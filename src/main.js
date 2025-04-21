import * as THREE from 'three';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';

// Scene setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ alpha: true }); // Enable transparency
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// --- Constants --- >
const HAND_CONFIDENCE = 0.5; // Minimum confidence score for hand detection/tracking
const INDEX_FINGER_TIP = 8; // Index for the tip of the index finger landmark
const SLICE_GESTURE_THRESHOLD = 0.03; // Minimum NDC distance for a valid slice - DECREASED for sensitivity

const FLOOR_Y = -2.5; // Define the ground level
const MAX_BLOOD_PARTICLES = 50; // Pool size
const INITIAL_SPAWN_INTERVAL = 2200; // Start at 2.2 seconds
const MIN_SPAWN_INTERVAL = 700; // Minimum interval (0.7 seconds)
const SPAWN_ACCELERATION = 0.995; // How quickly spawn interval decreases per spawn
let currentSpawnInterval = INITIAL_SPAWN_INTERVAL;
const ZOMBIE_START_Z = -20; // Start further away
const ZOMBIE_END_Z = 3;   // Point at which they stop/get removed closer to camera
const ZOMBIE_WALK_SPEED = 2.2; // Units per second - Faster zombies
const ZOMBIE_ANIM_FPS = 10; // Animation frames per second
const DAMPING = 0.95; // Damping factor for horizontal movement and rotation on floor
const ZOMBIE_FRAME_COUNT = 16;
const DESPAWN_TIME = 10.0; // Seconds until pieces/puddles despawn
// --- Perspective Constants (Needs Tuning!) --- >
const PERSPECTIVE_FLOOR_FACTOR = 0.015; // How much floor Y rises per unit of negative Z distance
const PATH_WIDTH_START = 8.0;     // Approx visual width of path at ZOMBIE_START_Z
const PATH_WIDTH_END = 2.0;       // Approx visual width of path at ZOMBIE_END_Z
const RED_LINE_Y_OFFSET = 0.0; // Offset of the "red line" spawn height relative to the calculated floor at spawn Z. Let's start with 0 (feet on floor) and adjust if needed.
// < --- Perspective Constants ---
// < --- Constants ---

// Texture Loading and Mesh Creation
const textureLoader = new THREE.TextureLoader();
const activeZombies = []; // Array to hold active zombie meshes
let zombieGeometry = null; // Store base geometry
const zombieFrames = []; // Array for animation frame textures
let allZombieFramesLoaded = false;

let textureSize = new THREE.Vector2(1, 1); // Default size (will be set by first frame)
const slicedPieces = [];
let isSliced = false;

// --- Hand Visualization --- >
const handGroup = new THREE.Group();
scene.add(handGroup);
const landmarkMeshes = [];
const LANDMARK_COUNT = 21;
// < --- Hand Visualization --- 

// --- Blood Splatter --- >
const bloodTextures = [];
const bloodParticlePool = [];
const activeBloodParticles = [];
let bloodTexturesLoaded = false;

const bloodTextureFiles = [
    '/sprites/blood/b1.png',
    '/sprites/blood/b2.png',
    '/sprites/blood/b3.png',
    '/sprites/blood/b4.png',
    '/sprites/blood/b5.png',
    '/sprites/blood/b6.png',
    '/sprites/blood/b7.png'
];

let loadedBloodCount = 0;
bloodTextureFiles.forEach(file => {
    textureLoader.load(file, 
        (texture) => {
            bloodTextures.push(texture);
            loadedBloodCount++;
            if (loadedBloodCount === bloodTextureFiles.length) {
                console.log('All blood textures loaded');
                bloodTexturesLoaded = true;
                // Initialize particle pool after textures are loaded
                for (let i = 0; i < MAX_BLOOD_PARTICLES; i++) {
                    const material = new THREE.SpriteMaterial({ 
                        map: bloodTextures[0], // Placeholder texture 
                        transparent: true, 
                        depthWrite: false, // Render on top
                        opacity: 0
                    });
                    const sprite = new THREE.Sprite(material);
                    sprite.visible = false; // Initially hidden
                    scene.add(sprite);
                    // Add velocity, rotationSpeed, and isOnFloor to particle data
                    bloodParticlePool.push({ 
                        sprite, 
                        lifetime: 0, 
                        velocity: new THREE.Vector2(), 
                        rotationSpeed: 0,
                        isOnFloor: false, 
                        despawnTimer: 0 // Initialize despawn timer
                    });
                }
            }
        },
        undefined,
        (error) => console.error(`Error loading blood texture ${file}:`, error)
    );
});
// < --- Blood Splatter ---

// Raycaster for accurate UV mapping
const raycaster = new THREE.Raycaster();

// --- Shaders ---

// Vertex Shader: Passes position and UVs
const vertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// Fragment Shader: Discards pixels based on slice line
const fragmentShader = `
  uniform sampler2D uTexture;
  uniform vec2 uSliceStart; // UV coordinates (0-1)
  uniform vec2 uSliceEnd;   // UV coordinates (0-1)
  uniform float uSideToKeep; // +1.0 or -1.0 (or 0.0 for no cut)

  varying vec2 vUv;

  void main() {
    // --- Slicing Logic --- >
    // Only perform slicing calculations if uSideToKeep is not 0.0
    if (uSideToKeep != 0.0) {
      // Handle cases where start and end are identical (or very close)
      if (distance(uSliceStart, uSliceEnd) < 0.0001) {
        // If points are the same, discard based on sign (arbitrary but consistent)
        if (uSideToKeep > 0.0) discard; 
      } else {
          // Line equation: ax + by + c = 0
          vec2 lineVec = uSliceEnd - uSliceStart;
          vec2 normal = normalize(vec2(-lineVec.y, lineVec.x)); // Normal vector
          vec2 pointVec = vUv - uSliceStart; // Vector from line start to current pixel
          float side = dot(pointVec, normal);

          // Discard pixel if it's on the wrong side
          if (side * uSideToKeep < 0.0) {
            discard;
          }
      }
    }
    // < --- Slicing Logic ---

    // Sample the texture
    vec4 texColor = texture2D(uTexture, vUv);

    // Discard transparent pixels from the texture itself
    if (texColor.a < 0.1) discard; 

    gl_FragColor = texColor;
  }
`;

// --- Load Zombie Animation Frames --- >
const zombieFrameFiles = [];
for (let i = 0; i < ZOMBIE_FRAME_COUNT; i++) {
    const frameNumber = i.toString().padStart(4, '0');
    // Construct filename based on the pattern observed
    zombieFrameFiles.push(`/sprites/zombie/1b70caea-08dc-45a4-8694-630ffc1193b5_angle_0_0_${frameNumber}.png`);
}

let loadedZombieFrameCount = 0;
zombieFrameFiles.forEach((file, index) => {
    textureLoader.load(file, 
        (texture) => {
            zombieFrames[index] = texture; // Store in order
            loadedZombieFrameCount++;
            if (index === 0) { // Use first frame to set geometry size
                textureSize.set(texture.image.width, texture.image.height);
                const aspectRatio = texture.image.width / texture.image.height;
                const planeHeight = 5; // Base height
                const planeWidth = planeHeight * aspectRatio;
                zombieGeometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
            }
            if (loadedZombieFrameCount === ZOMBIE_FRAME_COUNT) {
                console.log('All zombie animation frames loaded');
                allZombieFramesLoaded = true;
                // Start spawning only after frames AND geometry are ready
                startZombieSpawner(); 
            }
        },
        undefined,
        (error) => console.error(`Error loading zombie frame ${file}:`, error)
    );
});
// < --- Load Zombie Animation Frames ---

// --- Load Background --- >
textureLoader.load('/background/dungeon.png', 
    (texture) => {
        console.log('Background texture loaded successfully');
        scene.background = texture;
    },
    undefined,
    (error) => {
        console.error('Error loading background texture:', error);
    }
);
// < --- Load Background ---

// Camera position
camera.position.z = 5;

// Mouse tracking
const mouseNDC = new THREE.Vector2(); // Current mouse position in NDC
const sliceStartNDC = new THREE.Vector2(); // Where drag started in NDC
const sliceEndNDC = new THREE.Vector2(); // Where drag ended in NDC

// Store UVs *if* the ray hit the mesh at start/end
const sliceStartUV = new THREE.Vector2(); 
const sliceEndUV = new THREE.Vector2();
let sliceStartValid = false; // Did the mousedown hit the mesh?
let sliceEndValid = false;   // Did the mouseup hit the mesh?
let zombieToSlice = null; // Store the specific zombie hit on mousedown

let isSlicing = false;

// Helper function to get UV coords from mouse event
function getUVCoords(event) {
    // Ensure we have zombies to test against
    if (activeZombies.length === 0) {
        return null;
    }
    
    // Update mouseNDC based on event
    mouseNDC.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouseNDC.y = -(event.clientY / window.innerHeight) * 2 + 1; 
    
    // Update the picking ray with the camera and mouse position
    raycaster.setFromCamera(mouseNDC, camera);
    
    // Calculate objects intersecting the picking ray - test against ACTIVE zombies
    // Filter out zombies that might already be marked as sliced if needed
    const zombiesToTest = activeZombies.filter(z => !z.userData.isSliced); 
    const intersects = raycaster.intersectObjects(zombiesToTest); // Use intersectObjects
    
    if (intersects.length > 0) {
        // Return the UV coordinates AND the intersected object (zombie mesh)
        return {
            uv: intersects[0].uv, 
            object: intersects[0].object 
        };
    } else {
        // Return null if no intersection
        return null;
    }
}

// Handle mouse events
window.addEventListener('mousedown', (event) => {
    // Check if assets are loaded - replace isSliced check with zombie check later
    if (!zombieGeometry) return; 

    // Always record NDC start
    updateMousePosition(event); 
    sliceStartNDC.copy(mouseNDC);
    
    const hitData = getUVCoords(event); // Now returns { uv, object }
    if (hitData) {
        sliceStartUV.copy(hitData.uv);
        sliceStartValid = true;
        zombieToSlice = hitData.object; // Store the hit zombie
        isSlicing = true; // Only start slicing if we hit a zombie
    } else {
        sliceStartValid = false;
        zombieToSlice = null;
        isSlicing = false;
    }
    sliceEndValid = false; // Reset end validity
});

window.addEventListener('mousemove', (event) => {
    if (isSlicing) {
         updateMousePosition(event); 
         // Optional: Update visual feedback line during drag
    }
});

window.addEventListener('mouseup', (event) => {
    // Ensure we started slicing on a zombie
    if (isSlicing && zombieToSlice) { 
        updateMousePosition(event);
        sliceEndNDC.copy(mouseNDC);

        // We don't strictly need the end UV if we allow ending off-mesh
        // but we do need the end NDC for direction calculation
        const hitData = getUVCoords(event);
        if (hitData && hitData.object === zombieToSlice) { // Check if we ended on the *same* zombie
            sliceEndUV.copy(hitData.uv);
            sliceEndValid = true;
        } else {
            sliceEndValid = false; // Ended off-mesh or on a different zombie
        }

        // Only proceed if the drag was significant
        if (sliceStartNDC.distanceTo(sliceEndNDC) > 0.02) {
             // Pass the specific zombie to slice
            calculateAndPerformSlice(zombieToSlice); 
        }
    }
    // Reset state regardless of whether slice happened
    isSlicing = false;
    sliceStartValid = false; 
    sliceEndValid = false;
    zombieToSlice = null;
});

// Renamed from original updateMousePosition for clarity
function updateMousePosition(event) {
    mouseNDC.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouseNDC.y = -(event.clientY / window.innerHeight) * 2 + 1; 
}

// Handle window resize
window.addEventListener('resize', () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
});

// Function to intersect a ray with the unit UV square [0,1]x[0,1]
// Returns the intersection point or null if no intersection in the given direction
function intersectUVSquare(startPoint, direction) {
    let tMin = -Infinity;
    let tMax = Infinity;

    // Check intersection with vertical lines (x=0, x=1)
    if (Math.abs(direction.x) < 1e-6) { // Ray is vertical
        if (startPoint.x < 0 || startPoint.x > 1) return null; // Starts outside and parallel
    } else {
        let t1 = (0 - startPoint.x) / direction.x; // Intersection time with x=0
        let t2 = (1 - startPoint.x) / direction.x; // Intersection time with x=1
        if (t1 > t2) [t1, t2] = [t2, t1]; // Ensure t1 <= t2
        tMin = Math.max(tMin, t1);
        tMax = Math.min(tMax, t2);
    }

    // Check intersection with horizontal lines (y=0, y=1)
    if (Math.abs(direction.y) < 1e-6) { // Ray is horizontal
        if (startPoint.y < 0 || startPoint.y > 1) return null; // Starts outside and parallel
    } else {
        let t1 = (0 - startPoint.y) / direction.y; // Intersection time with y=0
        let t2 = (1 - startPoint.y) / direction.y; // Intersection time with y=1
        if (t1 > t2) [t1, t2] = [t2, t1]; // Ensure t1 <= t2
        tMin = Math.max(tMin, t1);
        tMax = Math.min(tMax, t2);
    }

    // If tMin > tMax, the ray misses the square entirely or goes backward through it
    if (tMin > tMax || tMax < 1e-6 ) { // tMax < epsilon handles cases where intersection is effectively the start point
        return null;
    }

    // We want the intersection point in the direction of the ray (smallest positive t, or tMin if positive)
    let tIntersection = (tMin >= 1e-6) ? tMin : tMax;
    
    // Check if the intersection time is reasonable (not excessively large)
    if (tIntersection < 1e-6 || tIntersection > 1e6) { 
        return null; // Avoid huge values if direction is near zero
    }
    
    return new THREE.Vector2().addVectors(startPoint, direction.clone().multiplyScalar(tIntersection));
}

// Modified to accept the specific zombie mesh
function calculateAndPerformSlice(targetZombie) { 
    if (!targetZombie) return; // Should not happen, but be safe

    let finalStartUV = new THREE.Vector2();
    let finalEndUV = new THREE.Vector2();
    
    // Calculate slice direction in NDC space
    const directionNDC = new THREE.Vector2().subVectors(sliceEndNDC, sliceStartNDC).normalize();
    // Approximate direction in UV space (simple mapping - might have perspective issues)
    // We scale Y because NDC Y range (-1 to 1) maps to UV Y range (0 to 1), while X ranges match up better.
    // This isn't perfect but better than unscaled.
    const directionUVApprox = new THREE.Vector2(directionNDC.x, directionNDC.y * (camera.aspect)).normalize(); 

    // Logic relies on sliceStartValid having hit the targetZombie
    if (sliceStartValid && sliceEndValid) {
        // Both points hit the target zombie
        finalStartUV.copy(sliceStartUV);
        finalEndUV.copy(sliceEndUV);
    } else if (sliceStartValid && !sliceEndValid) {
        // Start hit target, end missed: Extend line from startUV outwards
        finalStartUV.copy(sliceStartUV);
        const extendedEnd = intersectUVSquare(sliceStartUV, directionUVApprox);
        if (!extendedEnd) return; 
        finalEndUV.copy(extendedEnd);
    } 
    // Ignore case where start missed targetZombie - slicing must start on the zombie
    else { 
        console.warn("Invalid slice state for calculation.");
        return;
    }

    // Ensure the calculated points are distinct before slicing
    if (finalStartUV.distanceTo(finalEndUV) < 0.001) {
        console.warn("Slice points too close after extension.");
        return;
    }

    // Clamp UV coordinates to [0, 1] just in case of floating point errors
    finalStartUV.clampScalar(0, 1);
    finalEndUV.clampScalar(0, 1);

    // Pass the specific zombie and calculated UVs
    performSlice(targetZombie, finalStartUV, finalEndUV); 
}

// Modified performSlice to accept the target zombie and UVs
function performSlice(targetZombie, startUV, endUV) { 
    // Basic check (already sliced check below is more robust)
    if (!targetZombie || targetZombie.userData.isSliced) return;
    // --- KILL COOLDOWN ---
    const now = performance.now() / 1000;
    if (now - lastZombieKillTime < ZOMBIE_KILL_COOLDOWN) return;
    lastZombieKillTime = now;

    // 1. Get original mesh data
    const texture = targetZombie.material.uniforms.uTexture.value; // Get texture from material
    const geometry = targetZombie.geometry; // Use the specific zombie's geometry
    const originalPosition = targetZombie.position.clone();
    const originalRotation = targetZombie.rotation.clone();
    const originalScale = targetZombie.scale.clone();
    const planeWidth = geometry.parameters.width;
    const planeHeight = geometry.parameters.height;
    const originalMatrix = targetZombie.matrixWorld.clone(); 

    // 2. Remove original mesh from scene AND active list
    scene.remove(targetZombie);
    const index = activeZombies.indexOf(targetZombie);
    if (index > -1) {
        activeZombies.splice(index, 1);
    }
    // Don't dispose geometry IF we reuse the base zombieGeometry
    // Dispose material as it's unique per instance
    targetZombie.material.dispose(); 

    // 3. Create two new meshes with ShaderMaterial
    const createSlicedMesh = (sideToKeep) => {
        // Use the base geometry definition
        const newGeometry = zombieGeometry; // Reuse base geometry
        // OR: Clone if geometry was modified per zombie
        // const newGeometry = targetZombie.geometry.clone();

        const material = new THREE.ShaderMaterial({
            uniforms: {
                uTexture: { value: texture }, 
                uSliceStart: { value: startUV }, 
                uSliceEnd: { value: endUV },     
                uSideToKeep: { value: sideToKeep }
            },
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            transparent: true,
            side: THREE.DoubleSide 
        });

        const mesh = new THREE.Mesh(newGeometry, material); 
        // Apply original transform to the pieces
        mesh.position.copy(originalPosition);
        mesh.rotation.copy(originalRotation);
        mesh.scale.copy(originalScale); // Apply the scale of the sliced zombie
        return mesh;
    };

    const piece1 = createSlicedMesh(1.0); 
    const piece2 = createSlicedMesh(-1.0); 

    // 4. Add physics and slight separation 
    const sliceVectorUV = new THREE.Vector2().subVectors(endUV, startUV).normalize();
    const normalVectorWorld = new THREE.Vector2(-sliceVectorUV.y, sliceVectorUV.x).normalize(); 

    const forceMagnitude = 0.02 * originalScale.x; // Scale force by zombie size
    const rotationSpeed = 0.02; 
    const separation = 0.05 * originalScale.x; // Scale separation by zombie size

    piece1.position.x += normalVectorWorld.x * separation;
    piece1.position.y += normalVectorWorld.y * separation;
    piece1.userData.velocity = new THREE.Vector2(normalVectorWorld.x * forceMagnitude, normalVectorWorld.y * forceMagnitude);
    piece1.userData.rotation = rotationSpeed * (Math.random() > 0.5 ? 1 : -1); 
    piece1.userData.isOnFloor = false; 
    piece1.userData.despawnTimer = 0; // Initialize despawn timer

    piece2.position.x -= normalVectorWorld.x * separation;
    piece2.position.y -= normalVectorWorld.y * separation;
    piece2.userData.velocity = new THREE.Vector2(-normalVectorWorld.x * forceMagnitude, -normalVectorWorld.y * forceMagnitude);
    piece2.userData.rotation = rotationSpeed * (Math.random() > 0.5 ? 1 : -1);
    piece2.userData.isOnFloor = false; 
    piece2.userData.despawnTimer = 0; // Initialize despawn timer

    // 5. Add sliced pieces to scene and list
    scene.add(piece1);
    scene.add(piece2);
    slicedPieces.push(piece1, piece2);

    // --- Spawn Blood Splatter (using originalMatrix and dimensions) --- >
    if (bloodTexturesLoaded) {
        const numBloodParticles = 18;
        const centerUV = new THREE.Vector2().addVectors(startUV, endUV).multiplyScalar(0.5);
        const localX = (centerUV.x - 0.5) * planeWidth;
        const localY = (centerUV.y - 0.5) * planeHeight;
        // Use originalMatrix saved BEFORE removing the targetZombie
        const spawnPosition = new THREE.Vector3(localX, localY, 0).applyMatrix4(originalMatrix);

        for (let i = 0; i < numBloodParticles; i++) {
            // Get a particle from the pool
            const particleData = bloodParticlePool.find(p => !p.sprite.visible);
            if (!particleData) continue;

            const particle = particleData.sprite;
            particle.material.map = bloodTextures[Math.floor(Math.random() * bloodTextures.length)];
            particle.material.needsUpdate = true;
            particle.position.copy(spawnPosition);
            particle.position.z += (Math.random() - 0.5) * 0.1;
            particle.material.opacity = 1.0;
            particle.material.rotation = Math.random() * Math.PI * 2;
            const scale = (0.3 + Math.random() * 0.5) * originalScale.x; // Scale blood with zombie
            particle.scale.set(scale, scale, 1);
            particle.visible = true;

            // About 1/3 of the blood flies away in the cut direction (wall blood), rest falls to floor
            const isWallBlood = (i < Math.floor(numBloodParticles / 3));
            particleData.isWallBlood = isWallBlood;

            if (isWallBlood) {
                // Wall blood: flies in the cut direction, sticks to wall (ZOMBIE_START_Z)
                const cutAngle = Math.atan2(sliceVectorUV.y, sliceVectorUV.x);
                const angleOffset = (Math.random() - 0.5) * Math.PI * 0.25; // Less spread for wall blood
                const finalAngle = cutAngle + angleOffset;
                const speed = (0.09 + Math.random() * 0.08) * originalScale.x; // Faster for wall blood
                particleData.velocity = new THREE.Vector3(
                    Math.cos(finalAngle) * speed,
                    Math.sin(finalAngle) * speed,
                    -0.18 - Math.random() * 0.12 // Strong negative Z, toward wall
                );
                particleData.rotationSpeed = (Math.random() - 0.5) * 0.1;
                particleData.lifetime = 1.2 + Math.random() * 0.7;
                particleData.isOnFloor = false;
                particleData.despawnTimer = 0;
                particleData.stuckToWall = false;
            } else {
                // Floor blood: randomize direction, falls as before
                const angleOffset = (Math.random() - 0.5) * Math.PI * 0.8;
                const angle1 = Math.atan2(normalVectorWorld.y, normalVectorWorld.x) + angleOffset;
                const angle2 = Math.atan2(-normalVectorWorld.y, -normalVectorWorld.x) + angleOffset;
                const finalAngle = (Math.random() > 0.5) ? angle1 : angle2;
                const speed = (0.03 + Math.random() * 0.05) * originalScale.x;
                particleData.velocity = new THREE.Vector2(
                    Math.cos(finalAngle) * speed,
                    Math.sin(finalAngle) * speed
                );
                particleData.rotationSpeed = (Math.random() - 0.5) * 0.1;
                particleData.lifetime = 1.0 + Math.random() * 1.0;
                particleData.isOnFloor = false;
                particleData.despawnTimer = 0;
                particleData.stuckToWall = false;
            }
        }
    }
    // < --- Spawn Blood Splatter ---

    // --- SCORE: Add 50 points for slicing a zombie ---
    score += 50;
    updateScoreDisplay();
    playCutSound();

    // Note: isSliced is now handled per-zombie via userData
    // isSliced = true; // Remove global flag setting
}

// Animation loop
const clock = new THREE.Clock(); // Clock for delta time

// --- Audio Setup ---
const bgMusic = new Audio('/sounds/music.mp3');
bgMusic.loop = true;
bgMusic.volume = 0.5;
let musicStarted = false;

const cutSound = new Audio('/sounds/cut.mp3');
cutSound.volume = 0.7;

const ouchSound = new Audio('/sounds/ouch.mp3');
ouchSound.volume = 0.8;

const zombieSounds = [
    new Audio('/sounds/zom1.mp3'),
    new Audio('/sounds/zom2.mp3'),
    new Audio('/sounds/zom3.mp3')
];
zombieSounds.forEach(z => z.volume = 0.7);

function playCutSound() {
    cutSound.currentTime = 0;
    cutSound.play();
}
function playOuchSound() {
    ouchSound.currentTime = 0;
    ouchSound.play();
}
function playRandomZombieSound() {
    const idx = Math.floor(Math.random() * zombieSounds.length);
    const sound = zombieSounds[idx];
    sound.currentTime = 0;
    sound.play();
}

// Start music on first user interaction (required by browsers)
window.addEventListener('pointerdown', () => {
    if (!musicStarted) {
        bgMusic.play();
        musicStarted = true;
    }
}, { once: true });

// --- End Audio Setup ---

function animate() {
    requestAnimationFrame(animate);
    const deltaTime = clock.getDelta();

    // --- Move and Animate Active Zombies --- >
    const frameDuration = 1 / ZOMBIE_ANIM_FPS;
    for (let i = activeZombies.length - 1; i >= 0; i--) {
        const zombie = activeZombies[i];
        
        // --- Movement --- >
        if (zombie.userData.velocity) {
            // Update Z position first
            zombie.position.addScaledVector(zombie.userData.velocity, deltaTime);
            const currentZ = zombie.position.z;

            // --- Perspective Scaling --- > 
            const scaleFactor = Math.max(0.1, camera.position.z / (camera.position.z - currentZ)); 
            const currentScale = zombie.userData.baseScale * scaleFactor;
            zombie.scale.set(currentScale, currentScale, 1);
            
            // --- Update Y based on Perspective Floor --- >
            const currentFloorY = calculateFloorY(currentZ);
            const currentHeight = zombie.geometry.parameters.height * currentScale;
            zombie.position.y = currentFloorY + (currentHeight / 2);

            // --- Clamp X based on Perspective Path Width --- >
            const currentPathWidth = calculatePathWidth(currentZ);
            const halfWidth = currentPathWidth / 2;
            zombie.position.x = Math.max(-halfWidth, Math.min(halfWidth, zombie.position.x));

            // --- Removal Check --- >
            if (currentZ >= ZOMBIE_END_Z) { // Remove if too close
                scene.remove(zombie);
                zombie.material.dispose();
                activeZombies.splice(i, 1);
                // Lose a life and update hearts
                if (!gameOver && lives > 0) {
                    lives--;
                    updateHeartsDisplay();
                    flashScreenRed();
                    playOuchSound();
                    if (lives === 0) {
                        showGameOver();
                    }
                }
                continue; // Skip animation update for removed zombie
            }
        }
        // < --- Movement ---

        // --- Animation --- >
        zombie.userData.animTimer += deltaTime;
        if (zombie.userData.animTimer >= frameDuration) {
            zombie.userData.animTimer -= frameDuration; // Subtract duration, don't reset to 0
            zombie.userData.animFrame = (zombie.userData.animFrame + 1) % ZOMBIE_FRAME_COUNT;
            
            // Update texture uniform
            if (zombie.material.uniforms && zombie.material.uniforms.uTexture) {
                 zombie.material.uniforms.uTexture.value = zombieFrames[zombie.userData.animFrame];
            }
        }
        // < --- Animation ---
    }
    // < --- Move and Animate Active Zombies ---

    // Animate sliced pieces
    for (let i = slicedPieces.length - 1; i >= 0; i--) { 
        const piece = slicedPieces[i];
        if (!piece.userData.isOnFloor) { // Only apply physics if not on floor
            piece.position.x += piece.userData.velocity.x;
            piece.position.y += piece.userData.velocity.y;
            piece.userData.velocity.y -= 0.001; // Gravity
            piece.rotation.z += piece.userData.rotation;

            // Check for floor collision
            if (piece.position.y <= FLOOR_Y) {
                piece.position.y = FLOOR_Y; 
                piece.userData.isOnFloor = true;
                piece.userData.velocity.y = 0; 
                piece.userData.despawnTimer = 0; // Start despawn timer
            }
        } else {
            // Apply damping when on the floor
            piece.userData.velocity.x *= DAMPING;
            piece.userData.rotation *= DAMPING;
            // Move slightly based on damped velocity
            piece.position.x += piece.userData.velocity.x;
            piece.rotation.z += piece.userData.rotation;
            // Stop movement completely if slow enough
            if (Math.abs(piece.userData.velocity.x) < 0.001) piece.userData.velocity.x = 0;
            if (Math.abs(piece.userData.rotation) < 0.001) piece.userData.rotation = 0;

            // Increment despawn timer
            piece.userData.despawnTimer += deltaTime;
            if (piece.userData.despawnTimer >= DESPAWN_TIME) {
                // Despawn the piece
                scene.remove(piece);
                piece.geometry.dispose(); // Dispose geometry if it was unique
                piece.material.dispose();
                slicedPieces.splice(i, 1); // Remove from array
                continue; // Skip further processing for this piece
            }
        }
    }

    // --- Animate Blood Particles --- >
    if (bloodTexturesLoaded) {
        bloodParticlePool.forEach(particleData => {
            if (particleData.sprite.visible) {
                if (particleData.isWallBlood) {
                    // Wall blood logic
                    if (!particleData.stuckToWall) {
                        // Move in 3D (x, y, z)
                        particleData.sprite.position.x += particleData.velocity.x;
                        particleData.sprite.position.y += particleData.velocity.y;
                        particleData.sprite.position.z += particleData.velocity.z;
                        // Gravity (slight, so some arc)
                        particleData.velocity.y -= 0.003;
                        // Fade out as it flies
                        if (particleData.lifetime < 0.5) {
                            particleData.sprite.material.opacity = particleData.lifetime * 2;
                        }
                        // Stick to wall if z <= ZOMBIE_START_Z (background plane)
                        if (particleData.sprite.position.z <= ZOMBIE_START_Z) {
                            particleData.sprite.position.z = ZOMBIE_START_Z + 0.01 * (Math.random() - 0.5); // Slight random offset
                            particleData.velocity.set(0, 0, 0);
                            particleData.stuckToWall = true;
                            particleData.despawnTimer = 0;
                            particleData.sprite.material.opacity = 0.7 + Math.random() * 0.2;
                        }
                    } else {
                        // Stuck to wall: fade out over time
                        particleData.despawnTimer += deltaTime;
                        if (particleData.despawnTimer > 4.5) {
                            particleData.sprite.visible = false;
                            particleData.sprite.material.opacity = 0;
                            particleData.stuckToWall = false;
                        }
                    }
                    // Lifetime still decreases
                    particleData.lifetime -= deltaTime;
                    if (particleData.lifetime <= 0 && !particleData.stuckToWall) {
                        particleData.sprite.visible = false;
                        particleData.sprite.material.opacity = 0;
                    }
                } else if (!particleData.isOnFloor) {
                    // Floor blood logic (as before)
                    particleData.lifetime -= deltaTime;
                    if (particleData.lifetime <= 0) {
                        particleData.sprite.visible = false;
                        particleData.sprite.material.opacity = 0;
                    } else {
                        // Apply velocity and gravity
                        particleData.sprite.position.x += particleData.velocity.x;
                        particleData.sprite.position.y += particleData.velocity.y;
                        particleData.velocity.y -= 0.005; // Gravity for blood
                        // Apply rotation
                        particleData.sprite.material.rotation += particleData.rotationSpeed;
                        // Check for floor collision
                        if (particleData.sprite.position.y <= FLOOR_Y) {
                            particleData.sprite.position.y = FLOOR_Y + Math.random() * 0.01;
                            particleData.isOnFloor = true;
                            particleData.velocity.set(0, 0);
                            particleData.rotationSpeed = 0;
                            particleData.sprite.material.rotation = 0;
                            particleData.sprite.material.opacity = 0.6 + Math.random() * 0.2;
                            particleData.despawnTimer = 0;
                        } else {
                            // Fade out towards end of life only if airborne
                            if (particleData.lifetime < 0.5) {
                                particleData.sprite.material.opacity = particleData.lifetime * 2;
                            }
                        }
                    }
                } else {
                    // Is on floor (is a puddle)
                    particleData.despawnTimer += deltaTime;
                    if (particleData.despawnTimer >= DESPAWN_TIME) {
                        particleData.sprite.visible = false;
                        particleData.sprite.material.opacity = 0;
                        // Reset for pooling (optional but good practice)
                        particleData.isOnFloor = false;
                        particleData.despawnTimer = 0;
                    }
                }
            }
        });
    }
    // < --- Animate Blood Particles ---

    // --- Random zombie sounds when close to camera ---
    let lastZombieGroanTime = 0;
    function maybePlayZombieGroan(deltaTime) {
        const now = performance.now();
        if (now - lastZombieGroanTime < 2000) return; // At most every 2s
        for (const zombie of activeZombies) {
            if (zombie.position.z > ZOMBIE_END_Z - 2) { // Close to camera
                if (Math.random() < 0.03) { // 3% chance per frame if close
                    playRandomZombieSound();
                    lastZombieGroanTime = now;
                    break;
                }
            }
        }
    }
    maybePlayZombieGroan(deltaTime);

    renderer.render(scene, camera);
}

animate(); 

// --- Zombie Spawning ---
let spawnIntervalId = null;

function spawnZombie() {
    if (gameOver) return;
    // Ensure frames and geometry are ready
    if (!allZombieFramesLoaded || !zombieGeometry) return;

    // Create material with the first frame
    const material = new THREE.ShaderMaterial({ 
        uniforms: {
            uTexture: { value: zombieFrames[0] }, // Start with frame 0
            uSliceStart: { value: new THREE.Vector2(0, 0) }, 
            uSliceEnd: { value: new THREE.Vector2(0, 0) }, 
            uSideToKeep: { value: 0.0 } 
        },
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        transparent: true,
        side: THREE.DoubleSide 
    });

    const newZombie = new THREE.Mesh(zombieGeometry, material); 

    // Initial scale 
    const baseScale = 0.8 + Math.random() * 0.7;
    // Apply perspective scaling immediately based on start Z
    const initialScaleFactor = Math.max(0.1, camera.position.z / (camera.position.z - ZOMBIE_START_Z));
    const initialScale = baseScale * initialScaleFactor;
    newZombie.scale.set(initialScale, initialScale, 1);

    // Calculate starting position based on perspective
    const startPathWidth = calculatePathWidth(ZOMBIE_START_Z);
    const startX = (Math.random() - 0.5) * startPathWidth;
    // Calculate Y so feet are on the calculated floor + offset at START_Z
    const startFloorY = calculateFloorY(ZOMBIE_START_Z);
    const zombieHeight = newZombie.geometry.parameters.height * initialScale; // Use scaled height
    const startY = startFloorY + (zombieHeight / 2) + RED_LINE_Y_OFFSET;
    
    newZombie.position.set(startX, startY, ZOMBIE_START_Z);

    // Add animation and movement properties to userData
    newZombie.userData = {
        velocity: new THREE.Vector3(0, 0, ZOMBIE_WALK_SPEED), // Move only in Z
        isSliced: false, 
        animFrame: 0,
        animTimer: 0,
        baseScale: baseScale // Store base scale for perspective scaling
    };

    scene.add(newZombie);
    activeZombies.push(newZombie);

    // Accelerate spawn rate
    currentSpawnInterval = Math.max(MIN_SPAWN_INTERVAL, currentSpawnInterval * SPAWN_ACCELERATION);
    if (spawnIntervalId) {
        clearInterval(spawnIntervalId);
        spawnIntervalId = setInterval(spawnZombie, currentSpawnInterval);
    }
}

function startZombieSpawner() {
    if (spawnIntervalId) clearInterval(spawnIntervalId); // Clear existing interval if any
    currentSpawnInterval = INITIAL_SPAWN_INTERVAL;
    spawnZombie(); // Spawn one immediately
    spawnIntervalId = setInterval(spawnZombie, currentSpawnInterval);
}

function stopZombieSpawner() {
    if (spawnIntervalId) clearInterval(spawnIntervalId);
    spawnIntervalId = null;
}

// Call this somewhere appropriate if needed, e.g., game over
// stopZombieSpawner(); 

// --- Helper Functions for Perspective --- >
function calculateFloorY(z) {
    // Floor gets slightly higher further away (more negative z)
    // Base floor is FLOOR_Y, add offset based on distance from camera
    return FLOOR_Y + (camera.position.z - z) * PERSPECTIVE_FLOOR_FACTOR;
}

function calculatePathWidth(z) {
    // Linear interpolation of path width based on Z position
    const totalZDistance = ZOMBIE_END_Z - ZOMBIE_START_Z;
    // Ensure we don't divide by zero if start/end Z are the same
    if (Math.abs(totalZDistance) < 0.01) return PATH_WIDTH_END;
    
    const fraction = (z - ZOMBIE_START_Z) / totalZDistance;
    const width = THREE.MathUtils.lerp(PATH_WIDTH_START, PATH_WIDTH_END, fraction);
    return Math.max(0.1, width); // Ensure width is always positive
}
// < --- Helper Functions for Perspective ---> 

// --- Global Variables --- >
let handLandmarker = undefined;
let runningMode = "VIDEO"; // Or "IMAGE"
let webcamRunning = false;
const videoElement = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas"); // Get canvas element
const canvasCtx = canvasElement.getContext("2d");             // Get canvas context
let lastVideoTime = -1;
let handResults = undefined;
let enableWebcamButton;
let previousFingerTipNDC = null; // Track finger position across frames

// --- Slicing State (Now Hand-Based) --- >
// REMOVED: isHandSlicing, handSliceStartNDC, handSliceEndNDC, handSliceStartUV, handSliceEndUV, handSliceStartTime, SLICE_TIMEOUT, handZombieToSlice
// < --- Slicing State ---

// --- MediaPipe Hand Landmarker Setup --- >
const createHandLandmarker = async () => {
    console.log("[1] Creating Hand Landmarker...");
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
        );
        console.log("[2] Vision tasks resolver created.");
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
                delegate: "GPU"
            },
            runningMode: runningMode,
            numHands: 1, // Track only one hand for slicing
            minHandDetectionConfidence: HAND_CONFIDENCE,
            minHandPresenceConfidence: HAND_CONFIDENCE,
            minTrackingConfidence: HAND_CONFIDENCE
        });
        console.log("[3] Hand Landmarker instance created.");

        // --- Initialize 3D Hand Landmarks (Now a Line for Index Finger) --- >
        // const landmarkGeometry = new THREE.SphereGeometry(0.01); // Smaller spheres
        // const landmarkMaterial = new THREE.MeshBasicMaterial({ color: 0x00cccc }); // Cyan color
        // for (let i = 0; i < LANDMARK_COUNT; i++) {
        //     const mesh = new THREE.Mesh(landmarkGeometry, landmarkMaterial);
        //     mesh.visible = false; // Initially hidden
        //     handGroup.add(mesh);
        //     landmarkMeshes.push(mesh);
        // }
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0x00ffff, linewidth: 5 }); // Bright cyan line
        const points = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()]; // MCP, PIP, DIP, TIP
        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
        const indexFingerLine = new THREE.Line(lineGeometry, lineMaterial);
        indexFingerLine.visible = false;
        handGroup.add(indexFingerLine);
        // We'll reference indexFingerLine directly later, no need for landmarkMeshes array now
        // < --- Initialize 3D Hand Landmarks --- 

        // Now that vision tasks are ready, initialize Three.js related loaders/assets
        console.log("[4] Loading assets...");
        loadAssets(); // Load zombie frames, background, blood etc.
        
        // Add button to enable webcam
        console.log("[5] Adding webcam button...");
        addEnableWebcamButton();
        console.log("[6] Webcam button function called.");
    } catch (error) {
        console.error("Error during Hand Landmarker creation:", error);
        // Optionally display an error message to the user on the page
    }
};

function addEnableWebcamButton() {
    console.log("[7] Inside addEnableWebcamButton function.");
    enableWebcamButton = document.createElement('button');
    enableWebcamButton.id = 'enableWebcamButton'; // Assign ID for styling
    enableWebcamButton.textContent = 'ENABLE WEBCAM';
    enableWebcamButton.onclick = enableCam;
    document.body.appendChild(enableWebcamButton);
    console.log("[8] Webcam button appended to body.");
}

// --- Asset Loading (Moved Here) --- >
function loadAssets() {
    // Load Zombie Frames
    // ... (Keep existing zombie frame loading logic using textureLoader) ...
    const zombieFrameFiles = [];
    for (let i = 0; i < ZOMBIE_FRAME_COUNT; i++) {
        const frameNumber = i.toString().padStart(4, '0');
        zombieFrameFiles.push(`/sprites/zombie/1b70caea-08dc-45a4-8694-630ffc1193b5_angle_0_0_${frameNumber}.png`);
    }
    let loadedZombieFrameCount = 0;
    zombieFrameFiles.forEach((file, index) => {
        textureLoader.load(file, (texture) => {
             zombieFrames[index] = texture; 
             loadedZombieFrameCount++;
             if (index === 0) { 
                 textureSize.set(texture.image.width, texture.image.height);
                 const aspectRatio = texture.image.width / texture.image.height;
                 const planeHeight = 5; 
                 const planeWidth = planeHeight * aspectRatio;
                 zombieGeometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
             }
             if (loadedZombieFrameCount === ZOMBIE_FRAME_COUNT) {
                 console.log('All zombie animation frames loaded');
                 allZombieFramesLoaded = true;
                 // Start spawner only when frames and geometry are ready AND webcam enabled later
                 // startZombieSpawner(); // Moved to enableCam
             }
         }, undefined, (error) => console.error(`Error loading zombie frame ${file}:`, error));
    });

    // Load Background
    // ... (Keep existing background loading logic using textureLoader) ...
    textureLoader.load('/background/dungeon.png', (texture) => {
        console.log('Background texture loaded successfully');
        scene.background = texture;
    }, undefined, (error) => {
        console.error('Error loading background texture:', error);
    });

    // Load Blood Textures & Init Pool
    // ... (Keep existing blood loading and pooling logic using textureLoader) ...
     let loadedBloodCount = 0;
     bloodTextureFiles.forEach(file => {
         textureLoader.load(file, (texture) => {
             bloodTextures.push(texture);
             loadedBloodCount++;
             if (loadedBloodCount === bloodTextureFiles.length) {
                 console.log('All blood textures loaded');
                 bloodTexturesLoaded = true;
                 for (let i = 0; i < MAX_BLOOD_PARTICLES; i++) {
                     const material = new THREE.SpriteMaterial({ map: bloodTextures[0], transparent: true, depthWrite: false, opacity: 0 });
                     const sprite = new THREE.Sprite(material);
                     sprite.visible = false; 
                     scene.add(sprite);
                     bloodParticlePool.push({ sprite, lifetime: 0, velocity: new THREE.Vector2(), rotationSpeed: 0, isOnFloor: false, despawnTimer: 0 });
                 }
             }
         }, undefined, (error) => console.error(`Error loading blood texture ${file}:`, error));
     });
}
// < --- Asset Loading ---

// --- Webcam Handling --- >
function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! HandLandmarker not loaded yet.");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.textContent = "ENABLE WEBCAM";
        if(spawnIntervalId) stopZombieSpawner(); // Stop spawner if webcam disabled
        // Stop webcam stream?
        let stream = videoElement.srcObject;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        videoElement.srcObject = null;

    } else {
        webcamRunning = true;
        enableWebcamButton.textContent = "DISABLE WEBCAM";
        if(allZombieFramesLoaded && zombieGeometry) startZombieSpawner(); // Start spawner now

        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } }) // Prefer front camera
            .then((stream) => {
                videoElement.srcObject = stream;
                videoElement.addEventListener("loadeddata", predictWebcam);
            })
            .catch((err) => {
                console.error(err);
                webcamRunning = false;
                enableWebcamButton.textContent = "ENABLE WEBCAM";
            });
    }
}

// --- Prediction Loop --- >
let lastGestureTime = 0; // Can likely remove this later if unused
async function predictWebcam() {
    if (!webcamRunning) return;

    // Set canvas dimensions to match video
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    const startTimeMs = performance.now();
    if (lastVideoTime !== videoElement.currentTime && handLandmarker) {
        lastVideoTime = videoElement.currentTime;
        handResults = handLandmarker.detectForVideo(videoElement, startTimeMs);
    }

    // --- Draw landmarks --- >
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (handResults && handResults.landmarks) {
        for (const landmarks of handResults.landmarks) {
            // Draw connectors (lines between landmarks)
            drawConnectors(canvasCtx, landmarks, HandLandmarker.HAND_CONNECTIONS, {
                color: '#00FF00', // Green lines
                lineWidth: 5
            });
            // Draw landmarks (dots)
            drawLandmarks(canvasCtx, landmarks, { 
                color: '#FF0000', // Red dots
                lineWidth: 2 
            });
        }
    }
    canvasCtx.restore();
    // < --- Draw landmarks --- 

    // --- Update 3D Hand Visualization --- >
    if (handResults && handResults.landmarks && handResults.landmarks.length > 0) {
        handGroup.visible = true;
        const landmarks = handResults.landmarks[0]; // Use first detected hand
        const indexFingerIndices = [5, 6, 7, 8]; // MCP, PIP, DIP, TIP
        const points = [];
        let allIndexLandmarksVisible = true;

        for (let i = 0; i < indexFingerIndices.length; i++) {
            const landmarkIndex = indexFingerIndices[i];
            if (landmarks[landmarkIndex]) {
                const lm = landmarks[landmarkIndex];
                // Convert normalized (0-1) coords to NDC (-1 to 1)
                const ndcX = (lm.x * 2 - 1) * -1; 
                const ndcY = lm.y * -2 + 1; 
                const ndcZ = -0.5;
                
                const worldVec = new THREE.Vector3(ndcX, ndcY, ndcZ);
                worldVec.unproject(camera);
                points.push(worldVec);
            } else {
                allIndexLandmarksVisible = false;
                break; // Stop if any index landmark is missing
            }
        }

        // Update the line geometry if all points are valid
        const line = handGroup.children[0]; // Assumes line is the first child
        if (allIndexLandmarksVisible && line instanceof THREE.Line) {
            line.geometry.setFromPoints(points);
            line.geometry.attributes.position.needsUpdate = true; 
            line.visible = true;
        } else if (line) {
            line.visible = false;
        }
        
    } else {
        handGroup.visible = false; // Hide the whole group if no hands detected
    }
    // < --- Update 3D Hand Visualization --- 

    // Process hand results for slicing
    processHandData(handResults);

    // Call recursively
    window.requestAnimationFrame(predictWebcam);
}

// --- Gesture Recognition Helper --- >
// REMOVED: isIndexFingerPointing function
// < --- Gesture Recognition Helper --- 

// --- Gesture Processing & Slicing --- >
function processHandData(results) {
    if (gameOver) return;
    if (results && results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0]; // Use first detected hand
        const fingerTipRaw = landmarks[INDEX_FINGER_TIP]; // Normalized (0-1) video coords
        
        if (!fingerTipRaw) {
            // Finger tip lost, clear previous position
            previousFingerTipNDC = null; 
            return;
        }

        // Convert current landmark coords to NDC (-1 to 1), flipping Y
        const currentFingerTipNDC = new THREE.Vector2(
            (fingerTipRaw.x * 2 - 1) * -1, // Correct for mirrored view
            fingerTipRaw.y * -2 + 1      // Flip Y
        );

        // --- Frame-to-Frame Slicing Check --- >
        if (previousFingerTipNDC) {
            const distance = currentFingerTipNDC.distanceTo(previousFingerTipNDC);

            if (distance > SLICE_GESTURE_THRESHOLD) {
                // Potential slice: Check intersection from previous point
                const startHitData = getUVCoordsFromNDC(previousFingerTipNDC);
                
                if (startHitData && startHitData.object) { // Must start on a zombie
                    const targetZombie = startHitData.object;
                    const startUV = startHitData.uv;

                    // Check where the current point lands
                    const endHitData = getUVCoordsFromNDC(currentFingerTipNDC);
                    const endUVIfValid = (endHitData && endHitData.object === targetZombie) ? endHitData.uv : null;

                    // Perform the slice calculation based on this frame's movement
                    calculateAndPerformSliceHand(
                        targetZombie, 
                        startUV, 
                        previousFingerTipNDC, // Pass previous NDC
                        currentFingerTipNDC,  // Pass current NDC
                        endUVIfValid          // Pass current UV if valid
                    );
                }
            }
        }
        // < --- Frame-to-Frame Slicing Check --- 

        // Update previous position for the next frame
        previousFingerTipNDC = currentFingerTipNDC.clone();

    } else {
        // No hands detected, clear previous position
        previousFingerTipNDC = null; 
    }
}

// --- Updated Raycasting/Slicing Logic --- >

// Modified getUVCoords to accept NDC instead of event
function getUVCoordsFromNDC(ndcCoords) {
    if (activeZombies.length === 0) return null;
    raycaster.setFromCamera(ndcCoords, camera);
    const zombiesToTest = activeZombies.filter(z => !z.userData.isSliced); 
    const intersects = raycaster.intersectObjects(zombiesToTest); 
    if (intersects.length > 0) {
        return { uv: intersects[0].uv, object: intersects[0].object };
    } else {
        return null;
    }
}

// New function signature to handle slice calculation from frame-to-frame input
function calculateAndPerformSliceHand(targetZombie, startUV, startNDC, endNDC, endUVIfValid) {
    if (!targetZombie || targetZombie.userData.isSliced) return; // Added sliced check here too

    let finalStartUV = startUV.clone(); // Use the provided start UV
    let finalEndUV = new THREE.Vector2();

    // Calculate direction based on the NDC movement of this frame
    const directionNDC = new THREE.Vector2().subVectors(endNDC, startNDC).normalize();
    // Check for zero vector to prevent NaN issues
    if (directionNDC.lengthSq() < 1e-6) { 
        console.warn("Slice direction vector is zero.");
        return;
    }
    const directionUVApprox = new THREE.Vector2(directionNDC.x, directionNDC.y * camera.aspect).normalize();
    if (directionUVApprox.lengthSq() < 1e-6) { 
        console.warn("Approximate UV direction vector is zero.");
        return;
    }

    if (endUVIfValid) {
        // End point hit the target zombie
        finalEndUV.copy(endUVIfValid);
    } else {
        // End missed: Extend line from startUV outwards along the calculated direction
        const extendedEnd = intersectUVSquare(finalStartUV, directionUVApprox);
        if (!extendedEnd) {
             console.warn("Failed to extend slice line to UV edge."); 
             return; // Can't perform slice if extension fails
        }
        finalEndUV.copy(extendedEnd);
    }

    // Final check for point distance before performing the slice
    if (finalStartUV.distanceTo(finalEndUV) < 0.001) {
        console.warn("Final slice points too close after calculation/extension.");
        return;
    }

    // Clamp UV coordinates to [0, 1] just in case
    finalStartUV.clampScalar(0, 1);
    finalEndUV.clampScalar(0, 1);

    // Call the actual slicing function
    performSlice(targetZombie, finalStartUV, finalEndUV);
}

// --- Initialization --- >
createHandLandmarker(); // Start MediaPipe loading
// animate(); // Original animate() call (KEEP THIS ONE)

// ... (rest of functions: startZombieSpawner, stopZombieSpawner, calculateFloorY, calculatePathWidth) ... 
// ... (rest of functions: startZombieSpawner, stopZombieSpawner, calculateFloorY, calculatePathWidth) ... 

// --- Score System --- >
let score = 0;
let lives = 3;
let gameOver = false;
let lastZombieKillTime = 0;
const ZOMBIE_KILL_COOLDOWN = 0.2; // seconds

function updateScoreDisplay() {
    const el = document.getElementById('scoreDisplay');
    if (el) {
        el.textContent = score.toString().padStart(6, '0');
    }
}
function updateHeartsDisplay() {
    // Top heart is heart1, then heart2, then heart3 (top to bottom)
    for (let i = 1; i <= 3; i++) {
        const heart = document.getElementById('heart' + i);
        if (heart) {
            // heart1 is empty if lives < 3, heart2 if lives < 2, heart3 if lives < 1
            heart.src = (lives >= 4 - i) ? '/sprites/heart_full.png' : '/sprites/heart_empty.png';
        }
    }
}
function flashScreenRed() {
    let flash = document.createElement('div');
    flash.className = 'screen-flash';
    document.body.appendChild(flash);
    setTimeout(() => flash.remove(), 300);
}
function showGameOver() {
    gameOver = true;
    // Stop zombie spawner
    stopZombieSpawner();
    // Hide hearts and score
    document.getElementById('heartsContainer').style.display = 'none';
    document.getElementById('scoreDisplay').style.display = 'none';
    // Show overlay
    const overlay = document.getElementById('gameOverOverlay');
    overlay.style.display = 'flex';
    // Set final score
    const finalScore = document.getElementById('finalScore');
    if (finalScore) finalScore.textContent = score.toString().padStart(6, '0');
    // Stop webcam
    if (webcamRunning) enableCam();
}
// Play again button logic
if (typeof window !== 'undefined') {
    window.addEventListener('DOMContentLoaded', () => {
        const btn = document.getElementById('playAgainBtn');
        if (btn) btn.onclick = () => window.location.reload();
    });
}

// At the end of the file or after DOMContentLoaded, initialize the score and hearts display:
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        updateScoreDisplay();
        updateHeartsDisplay();
    });
} else {
    updateScoreDisplay();
    updateHeartsDisplay();
} 