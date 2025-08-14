import React, { useRef, useEffect, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  IconButton,
  Tooltip,
} from '@material-ui/core';
import {
  FullscreenExit,
  Fullscreen,
  CameraAlt,
  Settings,
  PlayArrow,
  Pause,
  VolumeUp,
} from '@material-ui/icons';
import { makeStyles } from '@material-ui/core/styles';
import * as THREE from 'three';

// Types
interface GDPDataPoint {
  country: string;
  period: string;
  gdp: number;
  x: number;
  y: number;
  z: number;
  color: string;
}

interface ThreeDVisualizationProps {
  data: GDPDataPoint[];
  width?: number;
  height?: number;
  onCountrySelect?: (country: string) => void;
  animate?: boolean;
}

const useStyles = makeStyles((theme) => ({
  container: {
    position: 'relative',
    width: '100%',
    height: '600px',
    overflow: 'hidden',
  },
  canvas: {
    display: 'block',
    width: '100%',
    height: '100%',
  },
  controls: {
    position: 'absolute',
    top: theme.spacing(1),
    right: theme.spacing(1),
    background: 'rgba(255, 255, 255, 0.9)',
    borderRadius: theme.spacing(1),
    padding: theme.spacing(1),
    zIndex: 1000,
  },
  timeline: {
    position: 'absolute',
    bottom: theme.spacing(1),
    left: theme.spacing(1),
    right: theme.spacing(1),
    background: 'rgba(255, 255, 255, 0.9)',
    borderRadius: theme.spacing(1),
    padding: theme.spacing(1),
    zIndex: 1000,
  },
  info: {
    position: 'absolute',
    top: theme.spacing(1),
    left: theme.spacing(1),
    background: 'rgba(0, 0, 0, 0.8)',
    color: 'white',
    borderRadius: theme.spacing(1),
    padding: theme.spacing(1),
    zIndex: 1000,
    minWidth: 200,
  },
}));

const ThreeDVisualization: React.FC<ThreeDVisualizationProps> = ({
  data,
  width = 800,
  height = 600,
  onCountrySelect,
  animate = false,
}) => {
  const classes = useStyles();
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const animationIdRef = useRef<number>();
  
  // State
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedCountry, setSelectedCountry] = useState<string>('');
  const [timeProgress, setTimeProgress] = useState(0);
  const [isAnimating, setIsAnimating] = useState(animate);
  const [visualizationType, setVisualizationType] = useState('spheres');
  const [colorScheme, setColorScheme] = useState('gdp');

  // Three.js objects
  const [spheres, setSpheres] = useState<THREE.Mesh[]>([]);
  const [labels, setLabels] = useState<THREE.Sprite[]>([]);

  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current) return;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000011);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(
      75,
      width / height,
      0.1,
      1000
    );
    camera.position.set(50, 50, 50);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true 
    });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    mountRef.current.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(50, 50, 50);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    // Grid
    const gridHelper = new THREE.GridHelper(100, 20, 0x888888, 0x444444);
    scene.add(gridHelper);

    // Axes helper
    const axesHelper = new THREE.AxesHelper(25);
    scene.add(axesHelper);

    // Controls (orbital controls)
    const controls = new (THREE as any).OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enableZoom = true;
    controls.enablePan = true;

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;
      
      const newWidth = mountRef.current.clientWidth;
      const newHeight = mountRef.current.clientHeight;
      
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [width, height]);

  // Update visualization when data changes
  useEffect(() => {
    if (!sceneRef.current || !data.length) return;

    // Clear existing objects
    spheres.forEach(sphere => sceneRef.current?.remove(sphere));
    labels.forEach(label => sceneRef.current?.remove(label));

    const newSpheres: THREE.Mesh[] = [];
    const newLabels: THREE.Sprite[] = [];

    // Create GDP visualization objects
    data.forEach((point, index) => {
      // Create sphere
      const geometry = new THREE.SphereGeometry(
        Math.log(point.gdp + 1) * 0.5, // Size based on GDP
        32,
        32
      );

      const material = new THREE.MeshPhongMaterial({
        color: new THREE.Color(point.color),
        transparent: true,
        opacity: 0.8,
      });

      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(point.x, point.y, point.z);
      sphere.castShadow = true;
      sphere.receiveShadow = true;
      sphere.userData = { country: point.country, gdp: point.gdp };

      sceneRef.current.add(sphere);
      newSpheres.push(sphere);

      // Create label
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d')!;
      context.font = '48px Arial';
      context.fillStyle = 'white';
      context.fillText(point.country, 0, 48);

      const texture = new THREE.CanvasTexture(canvas);
      const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.position.set(point.x, point.y + 5, point.z);
      sprite.scale.set(10, 5, 1);

      sceneRef.current.add(sprite);
      newLabels.push(sprite);
    });

    setSpheres(newSpheres);
    setLabels(newLabels);

    // Add mouse interaction
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onMouseClick = (event: MouseEvent) => {
      if (!mountRef.current || !cameraRef.current) return;

      const rect = mountRef.current.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, cameraRef.current);
      const intersects = raycaster.intersectObjects(newSpheres);

      if (intersects.length > 0) {
        const selectedObject = intersects[0].object;
        const country = selectedObject.userData.country;
        setSelectedCountry(country);
        onCountrySelect?.(country);

        // Highlight selected sphere
        newSpheres.forEach(sphere => {
          (sphere.material as THREE.MeshPhongMaterial).emissive.setHex(0x000000);
        });
        (selectedObject.material as THREE.MeshPhongMaterial).emissive.setHex(0x555555);
      }
    };

    if (mountRef.current) {
      mountRef.current.addEventListener('click', onMouseClick);
      return () => {
        mountRef.current?.removeEventListener('click', onMouseClick);
      };
    }
  }, [data, onCountrySelect]);

  // Animation timeline
  useEffect(() => {
    if (!isAnimating) return;

    const interval = setInterval(() => {
      setTimeProgress(prev => {
        const next = prev + 1;
        return next > 100 ? 0 : next;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isAnimating]);

  // Toggle fullscreen
  const toggleFullscreen = () => {
    if (!isFullscreen) {
      mountRef.current?.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
    setIsFullscreen(!isFullscreen);
  };

  // Take screenshot
  const takeScreenshot = () => {
    if (!rendererRef.current) return;
    
    const link = document.createElement('a');
    link.download = 'gdp-3d-visualization.png';
    link.href = rendererRef.current.domElement.toDataURL();
    link.click();
  };

  // Change visualization type
  const handleVisualizationTypeChange = (type: string) => {
    setVisualizationType(type);
    // Update visualization based on type
    // Implementation would vary based on requirements
  };

  return (
    <Paper className={classes.container}>
      <div ref={mountRef} className={classes.canvas} />
      
      {/* Controls */}
      <Box className={classes.controls}>
        <Grid container spacing={1} direction="column">
          <Grid item>
            <Tooltip title="Toggle Fullscreen">
              <IconButton size="small" onClick={toggleFullscreen}>
                {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
              </IconButton>
            </Tooltip>
            <Tooltip title="Take Screenshot">
              <IconButton size="small" onClick={takeScreenshot}>
                <CameraAlt />
              </IconButton>
            </Tooltip>
          </Grid>
          
          <Grid item>
            <FormControl size="small" style={{ minWidth: 120 }}>
              <InputLabel>Type</InputLabel>
              <Select
                value={visualizationType}
                onChange={(e) => handleVisualizationTypeChange(e.target.value as string)}
              >
                <MenuItem value="spheres">Spheres</MenuItem>
                <MenuItem value="bars">Bars</MenuItem>
                <MenuItem value="surface">Surface</MenuItem>
                <MenuItem value="particles">Particles</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item>
            <FormControl size="small" style={{ minWidth: 120 }}>
              <InputLabel>Color</InputLabel>
              <Select
                value={colorScheme}
                onChange={(e) => setColorScheme(e.target.value as string)}
              >
                <MenuItem value="gdp">GDP Value</MenuItem>
                <MenuItem value="growth">Growth Rate</MenuItem>
                <MenuItem value="region">Region</MenuItem>
                <MenuItem value="income">Income Level</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Box>

      {/* Timeline */}
      <Box className={classes.timeline}>
        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <IconButton
              size="small"
              onClick={() => setIsAnimating(!isAnimating)}
            >
              {isAnimating ? <Pause /> : <PlayArrow />}
            </IconButton>
          </Grid>
          
          <Grid item xs>
            <Slider
              value={timeProgress}
              onChange={(_, value) => setTimeProgress(value as number)}
              min={0}
              max={100}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => `${value}%`}
            />
          </Grid>
          
          <Grid item>
            <Typography variant="caption">
              Time Progress
            </Typography>
          </Grid>
        </Grid>
      </Box>

      {/* Information Panel */}
      {selectedCountry && (
        <Box className={classes.info}>
          <Typography variant="h6" gutterBottom>
            {selectedCountry}
          </Typography>
          <Typography variant="body2">
            GDP: ${data.find(d => d.country === selectedCountry)?.gdp.toLocaleString()} Billion
          </Typography>
          <Typography variant="body2">
            Period: {data.find(d => d.country === selectedCountry)?.period}
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default ThreeDVisualization;