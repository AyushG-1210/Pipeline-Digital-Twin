/**
 * Custom GLSL shaders for pipeline integrity visualization
 * Maps integrity values (0-1) to color gradients with PBR-style metallic rendering
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
  
  // Color mapping based on integrity — matches the app's benchmark palette
  vec3 getIntegrityColor(float integrity) {
    vec3 critical = vec3(0.839, 0.278, 0.235); // #D6473C
    vec3 warning = vec3(0.851, 0.557, 0.169);  // #D98E2B
    vec3 mild = vec3(0.298, 0.604, 0.471);     // #4C9A78
    vec3 healthy = vec3(0.408, 0.690, 0.671);  // #68B0AB
    
    if (integrity >= 0.8) {
      // Healthy range: interpolate between healthy and mild
      float t = (integrity - 0.8) / 0.2;
      return mix(mild, healthy, t);
    } else if (integrity >= 0.6) {
      // Mild degradation: interpolate between warning and mild
      float t = (integrity - 0.6) / 0.2;
      return mix(warning, mild, t);
    } else if (integrity >= 0.3) {
      // Warning range: interpolate between critical and warning
      float t = (integrity - 0.3) / 0.3;
      return mix(critical, warning, t);
    } else {
      // Critical range: stay red with slight darkening
      return critical * (0.7 + 0.3 * integrity / 0.3);
    }
  }
  
  void main() {
    // Base color from integrity
    vec3 baseColor = getIntegrityColor(uIntegrity);
    
    // Simple directional lighting
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diffuse = max(dot(vNormal, lightDir), 0.0);
    
    // Ambient + diffuse lighting
    float ambient = 0.4;
    float lighting = ambient + diffuse * 0.6;
    
    // Metallic effect
    vec3 viewDir = normalize(cameraPosition - vPosition);
    float fresnel = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 3.0);
    vec3 metallicHighlight = vec3(0.8) * fresnel * 0.3;
    
    // Final color with lighting
    vec3 finalColor = baseColor * lighting + metallicHighlight;
    
    // Selection highlight — a restrained pulsing rim glow in the accent color,
    // an outline rather than a full-surface tint.
    if (uSelected) {
      float pulse = 0.5 + 0.5 * sin(uTime * 3.0);
      vec3 accent = vec3(0.996, 0.298, 0.251); // #FE4C40
      finalColor += accent * fresnel * pulse * 0.9;
    }
    
    gl_FragColor = vec4(finalColor, 1.0);
  }
`;
