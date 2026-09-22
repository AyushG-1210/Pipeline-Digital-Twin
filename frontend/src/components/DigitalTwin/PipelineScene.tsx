import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei';
import Pipeline from './Pipeline';
import SoilOutlines from './SoilOutlines';
import SeverityScale from './SeverityScale';

/**
 * Main 3D scene containing the pipeline digital twin
 * Features orbital camera controls, industrial grid background, soil strata outlines, and severity scale legend
 */
export default function PipelineScene() {
    return (
        <div className="w-full h-full bg-[#0B1330] flex flex-col overflow-hidden">
            {/* 3D Canvas area */}
            <div className="flex-1 w-full min-h-0 relative bg-[#0B1330]">
                <Canvas>
                    {/* Background color — dark navy */}
                    <color attach="background" args={['#0B1330']} />

                    {/* Camera setup */}
                    <PerspectiveCamera makeDefault position={[15, 8, 20]} fov={45} />

                    {/* Orbital controls: Left-click rotates, Right-click pans */}
                    <OrbitControls
                        enableDamping
                        dampingFactor={0.05}
                        minDistance={1}
                        maxDistance={150}
                        maxPolarAngle={Math.PI / 2 + 0.05}
                    />

                    {/* Industrial Ambient + Key + Rim Lighting */}
                    <ambientLight intensity={0.5} />
                    <directionalLight position={[15, 20, 10]} intensity={1.0} castShadow />
                    <directionalLight position={[-15, 10, -10]} intensity={0.4} color="#68B0AB" />
                    <directionalLight position={[0, -10, 0]} intensity={0.25} color="#A5694F" />

                    {/* Primary Grid Floor - Soil Trench Reference */}
                    <Grid
                        args={[200, 200]}
                        position={[0, -3, 0]}
                        cellColor="#3E2D24"
                        sectionColor="#8D5B4C"
                        fadeDistance={140}
                        fadeStrength={1}
                    />

                    {/* Soil Strata and Trench Profile Outlines */}
                    <SoilOutlines />

                    {/* 3D Pipeline Digital Twin */}
                    <Pipeline />

                    {/* Depth Fog */}
                    <fog attach="fog" args={['#0B1330', 40, 180]} />
                </Canvas>
            </div>

            {/* Severity scale legend bar directly below the 3D representation */}
            <SeverityScale />
        </div>
    );
}


