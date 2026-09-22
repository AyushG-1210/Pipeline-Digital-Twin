/**
 * Custom GLSL shaders for pipeline integrity visualization
 * Maps integrity values (0-1) to color gradients with PBR metallic surface details
 * (weld seam accents, micro-texture, and specular highlights).
 */

export const vertexShader = `
  varying vec3 vNormal;
  varying vec3 vPosition;
  varying vec2 vUv;
  
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vPosition = position;
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const fragmentShader = `
  uniform float uIntegrity; // 0-1: 1 = healthy, 0 = critical
  uniform float uTime;
  uniform bool uSelected;
  
  varying vec3 vNormal;
  varying vec3 vPosition;
  varying vec2 vUv;
  
  // Color mapping based on integrity — matches benchmark palette
  vec3 getIntegrityColor(float integrity) {
    vec3 critical = vec3(0.839, 0.278, 0.235); // #D6473C
    vec3 warning = vec3(0.851, 0.557, 0.169);  // #D98E2B
    vec3 mild = vec3(0.298, 0.604, 0.471);     // #4C9A78
    vec3 healthy = vec3(0.408, 0.690, 0.671);  // #68B0AB
    
    if (integrity >= 0.8) {
      float t = (integrity - 0.8) / 0.2;
      return mix(mild, healthy, t);
    } else if (integrity >= 0.6) {
      float t = (integrity - 0.6) / 0.2;
      return mix(warning, mild, t);
    } else if (integrity >= 0.3) {
      float t = (integrity - 0.3) / 0.3;
      return mix(critical, warning, t);
    } else {
      return critical * (0.7 + 0.3 * integrity / 0.3);
    }
  }
  
  void main() {
    // Base color from integrity
    vec3 baseColor = getIntegrityColor(uIntegrity);

    // Industrial micro-texture: subtle metallic surface noise using UV harmonics
    float textureGrain = 0.96 + 0.08 * sin(vUv.x * 120.0) * cos(vUv.y * 60.0);
    baseColor *= textureGrain;

    // Circumferential Weld Seam Accents at segment ends and midpoint
    float weldSeam1 = smoothstep(0.02, 0.0, abs(vUv.y - 0.05));
    float weldSeam2 = smoothstep(0.02, 0.0, abs(vUv.y - 0.95));
    float weldSeamMid = smoothstep(0.015, 0.0, abs(vUv.y - 0.5));
    float weldIntensity = max(max(weldSeam1, weldSeam2), weldSeamMid);
    
    // Weld lines add a dark steel metallic seam accent
    vec3 weldColor = vec3(0.18, 0.22, 0.28);
    baseColor = mix(baseColor, weldColor, weldIntensity * 0.55);
    
    // Directional PBR lighting
    vec3 lightDir = normalize(vec3(1.0, 1.2, 0.8));
    float diffuse = max(dot(vNormal, lightDir), 0.0);
    
    // Ambient + key diffuse
    float ambient = 0.38;
    float lighting = ambient + diffuse * 0.55;
    
    // Metallic Fresnel & Specular Highlight
    vec3 viewDir = normalize(cameraPosition - vPosition);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(vNormal, halfDir), 0.0), 32.0);
    float fresnel = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 3.0);
    
    vec3 metallicHighlight = vec3(0.85, 0.9, 0.95) * (fresnel * 0.25 + spec * 0.4);
    
    // Final composite color
    vec3 finalColor = baseColor * lighting + metallicHighlight;
    
    // Selection pulsing rim highlight
    if (uSelected) {
      float pulse = 0.5 + 0.5 * sin(uTime * 3.5);
      vec3 accent = vec3(0.996, 0.298, 0.251); // #FE4C40
      finalColor += accent * (fresnel * 0.8 + 0.2) * pulse;
    }
    
    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

