import { useMemo } from 'react';
import * as THREE from 'three';

/**
 * Renders realistic soil layer outlines and trench excavation profile lines
 * surrounding the pipeline digital twin trajectory.
 * 
 * Features:
 * 1. Multi-tier soil strata wireframes (Topsoil, Sandy Loam, Clay, Bedrock).
 * 2. Dynamic U-trench excavation profile outlines sampled along the 3D pipeline curve.
 * 3. Longitudinal trench bedding guides.
 */
export default function SoilOutlines() {
    // Generate U-trench cross-section profile outlines and longitudinal guides
    const { profileLines, longitudinalLines, strataLines } = useMemo(() => {
        const SEGMENT_COUNT = 60;
        const SEGMENT_LENGTH = 2;
        const amplitudeY = 3;
        const frequencyY = 0.08;
        const amplitudeZ = 8;
        const frequencyZ = 0.12;

        const getPoint = (t: number): [number, number, number] => [
            t,
            Math.sin(t * frequencyY) * amplitudeY,
            Math.sin(t * frequencyZ) * amplitudeZ
        ];

        const getTangent = (t: number): [number, number, number] => [
            1,
            Math.cos(t * frequencyY) * frequencyY * amplitudeY,
            Math.cos(t * frequencyZ) * frequencyZ * amplitudeZ
        ];

        let currentT = -(SEGMENT_COUNT * SEGMENT_LENGTH) / 2;
        const pointsAlongCurve: { pos: THREE.Vector3; tangent: THREE.Vector3 }[] = [];

        for (let i = 0; i <= SEGMENT_COUNT; i++) {
            const p = getPoint(currentT);
            const tan = getTangent(currentT);
            pointsAlongCurve.push({
                pos: new THREE.Vector3(p[0], p[1], p[2]),
                tangent: new THREE.Vector3(tan[0], tan[1], tan[2]).normalize()
            });

            const tanLen = Math.sqrt(tan[0] * tan[0] + tan[1] * tan[1] + tan[2] * tan[2]);
            currentT += SEGMENT_LENGTH / tanLen;
        }

        // Trench dimensions relative to pipe centerline
        const trenchWidth = 2.4; // width X/Z perpendicular to pipe direction
        const trenchDepthBelow = 1.6; // distance below pipe centerline to trench bed
        const trenchHeightAbove = 1.8; // distance above pipe to topsoil trench lip

        const profilePoints: THREE.Vector3[] = [];
        const leftTopPoints: THREE.Vector3[] = [];
        const rightTopPoints: THREE.Vector3[] = [];
        const leftBottomPoints: THREE.Vector3[] = [];
        const rightBottomPoints: THREE.Vector3[] = [];

        // Sample U-trench profiles every 3 curve points
        pointsAlongCurve.forEach((pt, idx) => {
            const up = new THREE.Vector3(0, 1, 0);
            // Side vector perpendicular to tangent and up vector
            const side = new THREE.Vector3().crossVectors(pt.tangent, up).normalize();
            if (side.lengthSq() < 0.001) side.set(0, 0, 1);

            const halfW = trenchWidth / 2;
            const lt = pt.pos.clone().addScaledVector(side, -halfW).addScaledVector(up, trenchHeightAbove);
            const rt = pt.pos.clone().addScaledVector(side, halfW).addScaledVector(up, trenchHeightAbove);
            const lb = pt.pos.clone().addScaledVector(side, -halfW).addScaledVector(up, -trenchDepthBelow);
            const rb = pt.pos.clone().addScaledVector(side, halfW).addScaledVector(up, -trenchDepthBelow);

            leftTopPoints.push(lt);
            rightTopPoints.push(rt);
            leftBottomPoints.push(lb);
            rightBottomPoints.push(rb);

            // Add U-trench loop lines (LT -> LB -> RB -> RT -> LT)
            if (idx % 3 === 0) {
                profilePoints.push(lt, lb);
                profilePoints.push(lb, rb);
                profilePoints.push(rb, rt);
                profilePoints.push(rt, lt);
            }
        });

        // Create longitudinal trench boundary lines connecting profile loops
        const longPoints: THREE.Vector3[] = [];
        for (let i = 0; i < leftTopPoints.length - 1; i++) {
            longPoints.push(leftTopPoints[i], leftTopPoints[i + 1]);
            longPoints.push(rightTopPoints[i], rightTopPoints[i + 1]);
            longPoints.push(leftBottomPoints[i], leftBottomPoints[i + 1]);
            longPoints.push(rightBottomPoints[i], rightBottomPoints[i + 1]);
        }

        // Soil Strata Outlines (Layer boundary boxes around domain: X from -65 to +65, Z from -16 to +16)
        const strataPts: { pts: THREE.Vector3[]; color: string }[] = [];

        const createBoxOutline = (yMin: number, yMax: number, color: string) => {
            const minX = -65, maxX = 65;
            const minZ = -14, maxZ = 14;
            const pts: THREE.Vector3[] = [
                // Top loop
                new THREE.Vector3(minX, yMax, minZ), new THREE.Vector3(maxX, yMax, minZ),
                new THREE.Vector3(maxX, yMax, minZ), new THREE.Vector3(maxX, yMax, maxZ),
                new THREE.Vector3(maxX, yMax, maxZ), new THREE.Vector3(minX, yMax, maxZ),
                new THREE.Vector3(minX, yMax, maxZ), new THREE.Vector3(minX, yMax, minZ),
                // Bottom loop
                new THREE.Vector3(minX, yMin, minZ), new THREE.Vector3(maxX, yMin, minZ),
                new THREE.Vector3(maxX, yMin, minZ), new THREE.Vector3(maxX, yMin, maxZ),
                new THREE.Vector3(maxX, yMin, maxZ), new THREE.Vector3(minX, yMin, maxZ),
                new THREE.Vector3(minX, yMin, maxZ), new THREE.Vector3(minX, yMin, minZ),
                // Vertical posts
                new THREE.Vector3(minX, yMin, minZ), new THREE.Vector3(minX, yMax, minZ),
                new THREE.Vector3(maxX, yMin, minZ), new THREE.Vector3(maxX, yMax, minZ),
                new THREE.Vector3(maxX, yMin, maxZ), new THREE.Vector3(maxX, yMax, maxZ),
                new THREE.Vector3(minX, yMin, maxZ), new THREE.Vector3(minX, yMax, maxZ),
            ];
            strataPts.push({ pts, color });
        };

        // Topsoil Strata (+2m to +5m)
        createBoxOutline(2, 5, '#8D5B4C');
        // Subsoil/Clay Strata (-3m to +2m)
        createBoxOutline(-3, 2, '#7A5538');
        // Deep Bedrock Strata (-8m to -3m)
        createBoxOutline(-8, -3, '#4A3528');

        return {
            profileLines: profilePoints,
            longitudinalLines: longPoints,
            strataLines: strataPts
        };
    }, []);

    // Create line geometries
    const profileGeo = useMemo(() => {
        const geo = new THREE.BufferGeometry();
        geo.setFromPoints(profileLines);
        return geo;
    }, [profileLines]);

    const longGeo = useMemo(() => {
        const geo = new THREE.BufferGeometry();
        geo.setFromPoints(longitudinalLines);
        return geo;
    }, [longitudinalLines]);

    return (
        <group name="SoilOutlinesGroup">
            {/* Trench profile U-loops (Soil Cutaway outlines) */}
            <lineSegments geometry={profileGeo}>
                <lineBasicMaterial color="#C88A58" opacity={0.65} transparent linewidth={1.5} />
            </lineSegments>

            {/* Longitudinal Trench Bedding & Wall Edges */}
            <lineSegments geometry={longGeo}>
                <lineBasicMaterial color="#9E6B43" opacity={0.45} transparent linewidth={1} />
            </lineSegments>

            {/* Soil Strata Layer Boundary Boxes */}
            {strataLines.map((stratum, idx) => {
                const geo = new THREE.BufferGeometry().setFromPoints(stratum.pts);
                return (
                    <lineSegments key={idx} geometry={geo}>
                        <lineBasicMaterial color={stratum.color} opacity={0.3} transparent linewidth={1} />
                    </lineSegments>
                );
            })}
        </group>
    );
}
