import { useState, useEffect, useCallback, useRef, useMemo, Component } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid, PerspectiveCamera, Html } from '@react-three/drei'
import { Vector3, TubeGeometry, DoubleSide, CurvePath, LineCurve3 } from 'three'
import './App.css'

// Error boundary to catch Three.js crashes and show a useful message
class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }
  componentDidCatch(error, info) {
    console.error('SAFEGEN 3D Error:', error, info)
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          height: '100vh', background: '#0a0a18', color: '#ff4466',
          fontFamily: 'Inter, sans-serif', flexDirection: 'column', gap: 16,
        }}>
          <h2 style={{ color: '#e8e8f4' }}>Something went wrong</h2>
          <p style={{ color: '#8e8eb8', maxWidth: 500, textAlign: 'center' }}>
            {this.state.error?.message || 'Unknown error'}
          </p>
          <button
            onClick={() => window.location.reload()}
            style={{
              padding: '10px 24px', background: 'linear-gradient(135deg, #4d8aff, #00d4ff)',
              border: 'none', borderRadius: 8, color: 'white', fontSize: 14,
              fontWeight: 600, cursor: 'pointer',
            }}
          >
            Reload Page
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

// ─────────────────────────────────────────────────────────────────
// API Client
// ─────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000'

async function api(path, opts = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'API Error')
  }
  return res.json()
}

// ─────────────────────────────────────────────────────────────────
// 3D Components — Robust, animated, clear
// ─────────────────────────────────────────────────────────────────

function MazeWalls({ walls }) {
  if (!walls || walls.length === 0) return null
  return (
    <group>
      {walls.map((w, i) => (
        <mesh key={i} position={[w.x, 0.4, w.y]} castShadow receiveShadow>
          <boxGeometry args={[w.width * 0.95, 0.8, w.height * 0.95]} />
          <meshStandardMaterial
            color="#90a4ae"
            roughness={0.9}
            metalness={0.1}
          />
        </mesh>
      ))}
    </group>
  )
}

function Robot({ position, status }) {
  const groupRef = useRef()
  const targetPos = useRef(new Vector3(1.5, 0.35, 1.5))

  const y = position ? position[0] : 1.5
  const x = position ? position[1] : 1.5

  useFrame((_, delta) => {
    if (!groupRef.current) return
    targetPos.current.set(x, 0.25, y)
    groupRef.current.position.lerp(targetPos.current, Math.min(delta * 8, 1))
  })

  return (
    <group ref={groupRef} position={[x, 0.25, y]}>
      {/* Shadow */}
      <mesh position={[0, -0.24, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[0.6, 0.7]} />
        <meshBasicMaterial color="#000000" transparent opacity={0.3} />
      </mesh>
      
      {/* Base */}
      <mesh castShadow>
        <boxGeometry args={[0.5, 0.25, 0.6]} />
        <meshStandardMaterial color="#ffb300" roughness={0.6} metalness={0.2} />
      </mesh>
      
      {/* Scanner dome */}
      <mesh castShadow position={[0, 0.15, 0.1]}>
        <cylinderGeometry args={[0.1, 0.1, 0.15, 16]} />
        <meshStandardMaterial color="#212121" roughness={0.5} />
      </mesh>
      
      {/* Status light */}
      <mesh position={[0, 0.25, 0.1]}>
        <sphereGeometry args={[0.04, 16, 16]} />
        <meshBasicMaterial color={status === 'moving' ? '#00e676' : '#ff3d00'} />
      </mesh>

      <Html position={[0, 0.6, 0]} center distanceFactor={12} style={{ pointerEvents: 'none' }}>
        <div style={{
          background: 'rgba(255,255,255,0.9)',
          border: '1px solid #ffb300',
          borderRadius: 2,
          padding: '2px 4px',
          fontSize: 9,
          fontWeight: 800,
          color: '#333',
        }}>
          AGV-1
        </div>
      </Html>
    </group>
  )
}

function Marker3D({ position, color, label }) {
  if (!position) return null
  const y = position[0]
  const x = position[1]

  const isGoal = label === 'GOAL'
  const boxColor = isGoal ? '#4caf50' : '#2196f3'

  return (
    <group position={[x, 0.01, y]}>
      {/* Floor painted marking */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.3, 0.4, 32]} />
        <meshBasicMaterial color={boxColor} />
      </mesh>
      
      {/* Marker post */}
      <mesh castShadow position={[0, 0.25, 0]}>
        <boxGeometry args={[0.2, 0.5, 0.2]} />
        <meshStandardMaterial color={boxColor} roughness={0.7} />
      </mesh>

      <Html position={[0, 0.7, 0]} center distanceFactor={12} style={{ pointerEvents: 'none' }}>
        <div style={{
          background: boxColor,
          padding: '2px 6px',
          borderRadius: 2,
          fontSize: 10,
          fontWeight: 800,
          color: '#fff',
        }}>
          {label}
        </div>
      </Html>
    </group>
  )
}

function Obstacle3D({ obstacle }) {
  const meshRef = useRef()
  const isDynamic = obstacle.is_dynamic

  useFrame((_, delta) => {
    if (meshRef.current && isDynamic) {
      meshRef.current.rotation.y += delta * 1.5
    }
  })

  return (
    <group position={[obstacle.position[1], 0.0, obstacle.position[0]]}>
      {isDynamic ? (
        // Moving barrel
        <mesh ref={meshRef} position={[0, 0.4, 0]} castShadow>
          <cylinderGeometry args={[obstacle.radius * 0.7, obstacle.radius * 0.7, 0.8, 16]} />
          <meshStandardMaterial color="#ffea00" roughness={0.4} metalness={0.2} />
        </mesh>
      ) : (
        // Traffic Cone for static barrier
        <group position={[0, 0.4, 0]}>
          <mesh castShadow position={[0, -0.05, 0]}>
            <coneGeometry args={[obstacle.radius * 0.7, 0.7, 16]} />
            <meshStandardMaterial color="#ff5722" roughness={0.6} metalness={0.1} />
          </mesh>
          <mesh position={[0, 0.1, 0]}>
            <cylinderGeometry args={[obstacle.radius * 0.4, obstacle.radius * 0.5, 0.2, 16]} />
            <meshStandardMaterial color="#ffffff" roughness={0.6} />
          </mesh>
        </group>
      )}

      {/* Label */}
      <Html position={[0, 1.0, 0]} center distanceFactor={12} style={{ pointerEvents: 'none' }}>
        <div style={{
          background: 'rgba(255,255,255,0.9)',
          border: `1px solid ${isDynamic ? '#ffea00' : '#ff5722'}`,
          borderRadius: 2,
          padding: '2px 4px',
          fontSize: 8,
          fontWeight: 800,
          color: '#333',
        }}>
          {isDynamic ? '⚠ DYNAMIC' : 'CONE'}
        </div>
      </Html>
    </group>
  )
}

function TrajectoryTube({ points, color, type, yOffset = 0.15, showLabel = false }) {
  if (!points || points.length < 2) return null

  const isSafe = ['planned', 'executing', 'replanned'].includes(type)

  const curve = useMemo(() => {
    const pts = points.map(p => new Vector3(p[1], yOffset, p[0]))
    const path = new CurvePath()
    for (let i = 0; i < pts.length - 1; i++) {
      path.add(new LineCurve3(pts[i], pts[i + 1]))
    }
    return path
  }, [points, yOffset])

  const tubeGeo = useMemo(() => {
    const thickness = isSafe ? 0.04 : 0.005
    const segments = Math.min(points.length * 2, isSafe ? 150 : 50)
    return new TubeGeometry(curve, segments, thickness, isSafe ? 8 : 3, false)
  }, [curve, isSafe, points.length])

  const glowGeo = useMemo(() => {
    if (isSafe) {
      const segments = Math.min(points.length * 2, 150)
      return new TubeGeometry(curve, segments, 0.08, 8, false)
    }
    return null
  }, [curve, isSafe, points.length])

  const midPoint = useMemo(() => curve.getPointAt(0.5), [curve])

  const opacity = isSafe ? 1.0 : 0.15
  const emissiveIntensity = type === 'executing' ? 2.5 : isSafe ? 1.5 : 0.0

  return (
    <group>
      {/* Main trajectory */}
      <mesh geometry={tubeGeo}>
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={emissiveIntensity}
          transparent
          opacity={opacity}
          roughness={0.3}
          metalness={0.5}
        />
      </mesh>

      {/* Glow halo */}
      {glowGeo && (
        <mesh geometry={glowGeo}>
          <meshBasicMaterial color={color} transparent opacity={0.15} />
        </mesh>
      )}

      {/* Endpoint dots */}
      <mesh position={[points[0][1], yOffset, points[0][0]]}>
        <sphereGeometry args={[isSafe ? 0.06 : 0.02, 12, 12]} />
        <meshBasicMaterial color={color} transparent opacity={opacity} />
      </mesh>
      <mesh position={[points[points.length - 1][1], yOffset, points[points.length - 1][0]]}>
        <sphereGeometry args={[isSafe ? 0.06 : 0.02, 12, 12]} />
        <meshBasicMaterial color={color} transparent opacity={opacity} />
      </mesh>
      
      {/* Explicit Path Label */}
      {showLabel && (
        <Html position={[midPoint.x, midPoint.y + 0.5, midPoint.z]} center distanceFactor={15} style={{ pointerEvents: 'none', zIndex: 10 }}>
          <div style={{
            background: isSafe ? '#4caf50' : '#f44336',
            padding: '4px 8px',
            borderRadius: 4,
            fontSize: 12,
            fontWeight: 800,
            color: '#fff',
            whiteSpace: 'nowrap',
            boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
            border: '2px solid white'
          }}>
            {isSafe ? '✅ SAFE PATH' : '❌ UNSAFE PATH'}
          </div>
        </Html>
      )}
    </group>
  )
}

// ─────────────────────────────────────────────────────────────────
// Main 3D Scene
// ─────────────────────────────────────────────────────────────────

const TRAJ_COLORS = {
  planned: '#00FF66',
  executing: '#00E5FF',
  replanned: '#FFB300',
  rejected: '#FF3366',
  vanilla: '#FF5555',
}

function Scene3D({ state }) {
  const walls = state?.maze_walls || []
  const robot = state?.robot || { position: [1.5, 1.5], status: 'idle' }
  const obstacles = state?.obstacles || {}
  const trajectories = state?.trajectories || {}

  return (
    <Canvas shadows className="scene-canvas" gl={{ antialias: true }}>
      <color attach="background" args={['#e0e0e0']} />
      <fog attach="fog" args={['#e0e0e0', 12, 35]} />

      <PerspectiveCamera makeDefault position={[6, 11, 15]} fov={42} />
      <OrbitControls
        target={[6, 0, 4.5]}
        maxPolarAngle={Math.PI / 2.15}
        minPolarAngle={Math.PI / 8}
        minDistance={5}
        maxDistance={28}
        enableDamping
        dampingFactor={0.05}
      />

      {/* Lighting */}
      <ambientLight intensity={1.0} color="#ffffff" />
      <directionalLight
        position={[10, 20, 15]}
        intensity={2.0}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        color="#fff4e6"
      />
      <directionalLight position={[-10, 10, -10]} intensity={1.0} color="#e6f4ff" />

      {/* Floor */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[6, -0.01, 4.5]} receiveShadow>
        <planeGeometry args={[20, 15]} />
        <meshStandardMaterial color="#cfd8dc" roughness={0.8} metalness={0.1} />
      </mesh>

      {/* Grid */}
      <Grid
        position={[6, 0.0, 4.5]}
        args={[20, 15]}
        cellSize={1}
        cellThickness={0.5}
        cellColor="#b0bec5"
        sectionSize={4}
        sectionThickness={1}
        sectionColor="#90a4ae"
        fadeDistance={25}
        fadeStrength={1.5}
        infiniteGrid={false}
      />

      {/* Maze Walls */}
      <MazeWalls walls={walls} />

      {/* Obstacles */}
      {Object.values(obstacles).map(obs => (
        <Obstacle3D key={obs.id} obstacle={obs} />
      ))}

      {/* Trajectories */}
      {(() => {
        const allTrajs = Object.values(trajectories).sort((a, b) => (a.type === 'vanilla' ? -1 : 1))
        
        // Filter to only show EXACTLY 1 safe path
        const safeTraj = allTrajs.find(t => ['planned', 'executing', 'replanned'].includes(t.type))
        
        const filteredTrajs = []
        if (safeTraj) filteredTrajs.push(safeTraj)

        let markedSafe = false

        return filteredTrajs.map(traj => {
          const isSafe = true // Since we only pass safe trajectories now
          let showLabel = false
          if (isSafe && !markedSafe) {
            showLabel = true
            markedSafe = true
          }
          if (!isSafe && !markedUnsafe) {
            showLabel = true
            markedUnsafe = true
          }
          return (
            <TrajectoryTube
              key={traj.id}
              points={traj.points}
              color={TRAJ_COLORS[traj.type] || '#ffffff'}
              type={traj.type}
              yOffset={traj.type === 'vanilla' ? 0.22 : 0.15}
              showLabel={showLabel}
            />
          )
        })
      })()}

      {/* Markers */}
      <Marker3D position={state?.start} color="#00E5FF" label="START" />
      <Marker3D position={state?.goal} color="#FFB300" label="GOAL" />

      {/* Robot */}
      <Robot position={robot.position} status={robot.status} />
    </Canvas>
  )
}

// ─────────────────────────────────────────────────────────────────
// UI Components
// ─────────────────────────────────────────────────────────────────

function StatusBadge({ status }) {
  const colors = {
    idle: 'linear-gradient(135deg, #4b5563, #6b7280)',
    planning: 'linear-gradient(135deg, #d97706, #f59e0b)',
    executing: 'linear-gradient(135deg, #2563eb, #3b82f6)',
    paused: 'linear-gradient(135deg, #d97706, #f59e0b)',
    replanning: 'linear-gradient(135deg, #7c3aed, #a855f7)',
    completed: 'linear-gradient(135deg, #059669, #10b981)',
    failed: 'linear-gradient(135deg, #dc2626, #ef4444)',
  }
  return (
    <span className="status-badge" style={{ background: colors[status] || colors.idle }}>
      {status?.toUpperCase() || 'UNKNOWN'}
    </span>
  )
}

function getWorkflowStep(state) {
  if (!state?.start || !state?.goal) return 1
  const hasTraj = Object.keys(state?.trajectories || {}).length > 0
  if (!hasTraj) return 2
  if (state?.status === 'executing') return 4
  if (state?.status === 'completed') return 5
  return 3
}

function MissionPanel({ state, onCreateMission, onGenerate, onInjectObstacle, onStartExec, onPause, onReset, onReplan }) {
  const [startY, setStartY] = useState('1.5')
  const [startX, setStartX] = useState('1.5')
  const [goalY, setGoalY] = useState('7.5')
  const [goalX, setGoalX] = useState('10.5')
  const [plannerMode, setPlannerMode] = useState('safe_diffuser')
  const [perfMode, setPerfMode] = useState('balanced')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const currentStep = getWorkflowStep(state)

  const handleCreate = async () => {
    setError('')
    try {
      await onCreateMission([parseFloat(startY), parseFloat(startX)], [parseFloat(goalY), parseFloat(goalX)])
    } catch (e) { setError(e.message) }
  }

  const handleGenerate = async () => {
    setLoading(true)
    setError('')
    try {
      if (!state?.start || !state?.goal) {
        await onCreateMission([parseFloat(startY), parseFloat(startX)], [parseFloat(goalY), parseFloat(goalX)])
      }
      await onGenerate(plannerMode, perfMode)
    } catch (e) { setError(e.message) }
    setLoading(false)
  }

  return (
    <div className="panel mission-panel">
      <h3>🎯 Mission Control</h3>

      <div className="guide-tip">
        <span className="guide-tip-icon">💡</span>
        <span className="guide-tip-text">
          Follow the steps below to <strong>plan and execute</strong> a safe trajectory through the maze.
        </span>
      </div>

      {/* Step 1 */}
      <div className={`workflow-section animate-in ${currentStep === 1 ? 'active' : currentStep > 1 ? 'completed' : ''}`}>
        <div className="step-header">
          <span className="step-number">{currentStep > 1 ? '✓' : '1'}</span>
          <span className="step-title">Set Start & Goal</span>
        </div>
        <div className="input-group">
          <label>Start Position (Y, X)</label>
          <div className="input-row">
            <input type="number" value={startY} onChange={e => setStartY(e.target.value)} step="0.5" min="0.5" max="8.5" placeholder="Y" />
            <input type="number" value={startX} onChange={e => setStartX(e.target.value)} step="0.5" min="0.5" max="11.5" placeholder="X" />
          </div>
        </div>
        <div className="input-group">
          <label>Goal Position (Y, X)</label>
          <div className="input-row">
            <input type="number" value={goalY} onChange={e => setGoalY(e.target.value)} step="0.5" min="0.5" max="8.5" placeholder="Y" />
            <input type="number" value={goalX} onChange={e => setGoalX(e.target.value)} step="0.5" min="0.5" max="11.5" placeholder="X" />
          </div>
        </div>
        <button className="btn btn-primary" onClick={handleCreate}>📍 Set Mission</button>
      </div>

      {/* Step 2 */}
      <div className={`workflow-section animate-in ${currentStep === 2 ? 'active' : currentStep > 2 ? 'completed' : ''}`}>
        <div className="step-header">
          <span className="step-number">{currentStep > 2 ? '✓' : '2'}</span>
          <span className="step-title">Generate Trajectory</span>
        </div>
        <div className="input-group">
          <label>Planner Mode</label>
          <select value={plannerMode} onChange={e => setPlannerMode(e.target.value)}>
            <option value="safe_diffuser">🛡️ SafeDiffuser (Safe)</option>
            <option value="vanilla">⚡ Vanilla Diffuser</option>
          </select>
        </div>
        <div className="input-group">
          <label>Performance</label>
          <select value={perfMode} onChange={e => setPerfMode(e.target.value)}>
            <option value="fast_demo">🚀 Fast Demo</option>
            <option value="balanced">⚖️ Balanced</option>
            <option value="research">🔬 Research Quality</option>
          </select>
        </div>
        <button className="btn btn-accent" onClick={handleGenerate} disabled={loading}>
          {loading ? (<><span className="spinner" /> Planning...</>) : '🚀 Generate Trajectory'}
        </button>
      </div>

      {/* Step 3 */}
      <div className={`workflow-section animate-in ${currentStep >= 3 ? 'active' : ''}`}>
        <div className="step-header">
          <span className="step-number">3</span>
          <span className="step-title">Execute & Interact</span>
        </div>
        <div className="button-group">
          <button className="btn btn-success" onClick={onStartExec}
            disabled={!Object.keys(state?.trajectories || {}).length || state?.status === 'executing'}>
            ▶ Execute
          </button>
          <button className="btn btn-warn" onClick={onPause} disabled={state?.status !== 'executing'}>
            ⏸ Pause
          </button>
        </div>
        <button className="btn btn-danger" onClick={onInjectObstacle}>⚠️ Inject Obstacle</button>
        <button className="btn btn-secondary" onClick={onReplan}>🔄 Replan Route</button>
      </div>

      <div className="separator" />
      <button className="btn btn-ghost" onClick={onReset}>↺ Reset Everything</button>

      {error && <div className="error-msg">❌ {error}</div>}
    </div>
  )
}

function StatusPanel({ state }) {
  const trajectories = Object.values(state?.trajectories || {})
  const latestTraj = trajectories.length > 0 ? trajectories[trajectories.length - 1] : null
  const latestMetrics = latestTraj?.metrics || {}

  const trajColors = {
    planned: '#00FF66', executing: '#00E5FF', replanned: '#FFB300',
    rejected: '#FF3366', vanilla: '#FF5555',
  }

  return (
    <div className="panel status-panel">
      <h3>📊 AI Planner Status</h3>

      <div className="stat-group animate-in">
        <div className="stat">
          <span className="stat-label">Diffusion Status</span>
          <StatusBadge status={state?.status} />
        </div>
        <div className="stat">
          <span className="stat-label">Safety Verdict</span>
          <span className={`stat-value ${latestTraj?.is_safe === true ? 'safe' : latestTraj?.is_safe === false ? 'unsafe' : ''}`}>
            {latestTraj?.is_safe === true ? '✅ SAFE' : latestTraj?.is_safe === false ? '❌ UNSAFE' : '—'}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Safety Violations</span>
          <span className={`stat-value ${(latestMetrics?.safety_violations || 0) > 0 ? 'unsafe' : ''}`}>
            {latestMetrics?.safety_violations ?? '—'}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Min Safety Margin</span>
          <span className={`stat-value ${(latestMetrics?.min_safety_margin || 0) < 0 ? 'unsafe' : ''}`}>
            {latestMetrics?.min_safety_margin?.toFixed(3) ?? '—'}
          </span>
        </div>
      </div>

      <div className="separator" />

      <h4>Trajectories</h4>
      <div className="traj-section">
        {trajectories.length === 0 ? (
          <div className="empty-state">No trajectories yet. Generate a plan to begin.</div>
        ) : (
          trajectories.map(traj => (
            <div key={traj.id} className="traj-item" style={{ borderLeftColor: trajColors[traj.type] || '#666' }}>
              <span className="traj-type">{traj.type}</span>
              <span className="traj-safety">{traj.is_safe ? '✅' : '❌'}</span>
            </div>
          ))
        )}
      </div>

      <div className="separator" />

      <h4>Color Legend</h4>
      <div className="legend">
        <div className="legend-item"><span className="legend-dot" style={{ background: '#00FF66' }} /> Safe Path</div>
        <div className="legend-item"><span className="legend-dot" style={{ background: '#FF5555' }} /> Unsafe Path</div>
        <div className="legend-item"><span className="legend-dot" style={{ background: '#00E5FF' }} /> Executing</div>
        <div className="legend-item"><span className="legend-dot" style={{ background: '#FFB300' }} /> Replanned</div>
        <div className="legend-item"><span className="legend-dot" style={{ background: '#00E5FF' }} /> Start Point</div>
        <div className="legend-item"><span className="legend-dot" style={{ background: '#FFB300' }} /> Goal Point</div>
        <div className="legend-item"><span className="legend-dot" style={{ background: '#FF3366' }} /> Static Barrier</div>
        <div className="legend-item"><span className="legend-dot" style={{ background: '#FF6E00' }} /> Dynamic Obs.</div>
      </div>

      <div className="separator" />

      <h4>Event Log</h4>
      <div className="events-list">
        {(state?.events || []).length === 0 ? (
          <div className="empty-state">No events recorded yet.</div>
        ) : (
          (state?.events || []).slice(-10).reverse().map(evt => (
            <div key={evt.id} className="event-item">
              <span className="event-dot" />
              <span className="event-type">{evt.type}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

function MetricsBar({ state }) {
  const trajectories = Object.values(state?.trajectories || {})
  const latest = trajectories.length > 0 ? trajectories[trajectories.length - 1].metrics : {}

  return (
    <div className="metrics-bar">
      <div className="metric">
        <span className="metric-label">Planning Time</span>
        <span className="metric-value">{latest?.planning_time_ms?.toFixed(0) ?? '—'} ms</span>
      </div>
      <div className="metric">
        <span className="metric-label">Trajectory Length</span>
        <span className="metric-value">{latest?.trajectory_length?.toFixed(2) ?? '—'}</span>
      </div>
      <div className="metric">
        <span className="metric-label">Safety Violations</span>
        <span className={`metric-value ${(latest?.safety_violations || 0) > 0 ? 'danger' : 'good'}`}>
          {latest?.safety_violations ?? '—'}
        </span>
      </div>
      <div className="metric">
        <span className="metric-label">Min Safety Margin</span>
        <span className={`metric-value ${(latest?.min_safety_margin || 0) < 0 ? 'danger' : 'good'}`}>
          {latest?.min_safety_margin?.toFixed(4) ?? '—'}
        </span>
      </div>
      <div className="metric">
        <span className="metric-label">Replans</span>
        <span className="metric-value">{state?.replan_count ?? 0}</span>
      </div>
      <div className="metric">
        <span className="metric-label">Progress</span>
        <span className="metric-value">{((state?.execution_progress || 0) * 100).toFixed(0)}%</span>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────

function App() {
  const [state, setState] = useState({
    robot: { position: [1.5, 1.5], status: 'idle' },
    obstacles: {},
    trajectories: {},
    goal: [7.5, 10.5],
    start: [1.5, 1.5],
    status: 'idle',
    replan_count: 0,
    execution_progress: 0,
    events: [],
    maze_walls: [],
  })
  const [connected, setConnected] = useState(false)
  const [planning, setPlanning] = useState(false)
  const wsRef = useRef(null)

  // Load initial data
  useEffect(() => {
    api('/environment').then(env => {
      setState(prev => ({ ...prev, maze_walls: env.walls }))
    }).catch(() => {})
    api('/simulation/state').then(st => {
      setState(prev => ({ ...prev, ...st }))
    }).catch(() => {})
  }, [])

  // Periodic sync fallback
  useEffect(() => {
    const interval = setInterval(() => {
      api('/simulation/state').then(st => {
        setState(prev => ({ ...prev, ...st }))
      }).catch(() => {})
    }, 1500)
    return () => clearInterval(interval)
  }, [])

  // WebSocket
  useEffect(() => {
    let ws
    let reconnectTimeout

    const connect = () => {
      ws = new WebSocket('ws://localhost:8000/ws/simulation')
      wsRef.current = ws

      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        reconnectTimeout = setTimeout(connect, 3000)
      }
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data)
          if (data.type !== 'pong') {
            setState(prev => ({ ...prev, ...data }))
          }
        } catch {}
      }
    }

    connect()
    return () => {
      clearTimeout(reconnectTimeout)
      ws?.close()
    }
  }, [])

  // Handlers
  const handleCreateMission = useCallback(async (start, goal) => {
    await api('/planning/create', { method: 'POST', body: JSON.stringify({ start, goal }) })
    const st = await api('/simulation/state')
    setState(prev => ({ ...prev, ...st }))
  }, [])

  const handleGenerate = useCallback(async (plannerMode, perfMode) => {
    setPlanning(true)
    try {
      if (!state.start || !state.goal) {
        await api('/planning/create', { method: 'POST', body: JSON.stringify({ start: [1.5, 1.5], goal: [7.5, 10.5] }) })
      }
      await api('/planning/generate', {
        method: 'POST',
        body: JSON.stringify({ planner_mode: plannerMode, performance_mode: perfMode, batch_size: 4 }),
      })
      const st = await api('/simulation/state')
      setState(prev => ({ ...prev, ...st }))
    } finally {
      setPlanning(false)
    }
  }, [state.start, state.goal])

  const handleInjectObstacle = useCallback(async () => {
    let pos = [5.5, 5.5]
    const trajs = Object.values(state?.trajectories || {})
    const activeTraj = trajs.find(t => t.type === 'executing' || t.type === 'planned' || t.type === 'replanned')
    if (activeTraj?.points?.length > 10) {
      const idx = Math.floor(activeTraj.points.length * 0.55)
      pos = [parseFloat(activeTraj.points[idx][0].toFixed(2)), parseFloat(activeTraj.points[idx][1].toFixed(2))]
    } else {
      const positions = [[4.5, 3.5], [3.5, 7.5], [5.5, 5.5], [6.5, 4.5], [2.5, 9.5]]
      pos = positions[Math.floor(Math.random() * positions.length)]
    }
    await api('/simulation/obstacle', { method: 'POST', body: JSON.stringify({ position: pos, radius: 0.8, label: 'Dynamic Obstacle' }) })
    const st = await api('/simulation/state')
    setState(prev => ({ ...prev, ...st }))
  }, [state?.trajectories])

  const handleStartExec = useCallback(async () => {
    await api('/simulation/start', { method: 'POST' })
  }, [])

  const handlePause = useCallback(async () => {
    await api('/simulation/pause', { method: 'POST' })
    const st = await api('/simulation/state')
    setState(prev => ({ ...prev, ...st }))
  }, [])

  const handleReset = useCallback(async () => {
    await api('/simulation/reset', { method: 'POST' })
    const st = await api('/simulation/state')
    setState(prev => ({ ...prev, ...st }))
  }, [])

  const handleReplan = useCallback(async () => {
    await api('/planning/replan', { method: 'POST' })
    const st = await api('/simulation/state')
    setState(prev => ({ ...prev, ...st }))
  }, [])

  const isExecuting = state.status === 'executing'
  const showProgress = state.execution_progress > 0 && state.execution_progress < 1 && isExecuting

  return (
    <div className="app">
      {/* Connection warning */}
      {!connected && (
        <div className="connection-banner">
          <span className="spinner" />
          Connecting to backend server...
        </div>
      )}

      {/* Top Bar */}
      <header className="top-bar">
        <div className="logo">
          <span className="logo-icon">◈</span>
          <h1>SAFEGEN 3D</h1>
          <span className="subtitle">Safe Generative Planning & Digital Twin</span>
        </div>
        <div className="top-status">
          <div className="status-item">
            <span className="dot" style={{ background: connected ? '#00f088' : '#ff4466', color: connected ? '#00f088' : '#ff4466' }} />
            <span>{connected ? 'Live' : 'Offline'}</span>
          </div>
          <StatusBadge status={state.status} />
        </div>
      </header>

      {/* Main Layout */}
      <div className="main-layout">
        <MissionPanel
          state={state}
          onCreateMission={handleCreateMission}
          onGenerate={handleGenerate}
          onInjectObstacle={handleInjectObstacle}
          onStartExec={handleStartExec}
          onPause={handlePause}
          onReset={handleReset}
          onReplan={handleReplan}
        />

        <div className="scene-container">
          <Scene3D state={state} />

          {/* Camera controls help */}
          <div className="scene-controls-help">
            <span><kbd>Drag</kbd> Rotate view</span>
            <span><kbd>Scroll</kbd> Zoom in/out</span>
            <span><kbd>Right-drag</kbd> Pan</span>
          </div>

          {/* Planning overlay */}
          {planning && (
            <div className="progress-overlay">
              <div className="spinner-large" />
              <div className="progress-overlay-title">Generating Trajectory</div>
              <div className="progress-overlay-sub">SafeDiffuser is computing optimal paths...</div>
            </div>
          )}

          {/* Execution progress bar */}
          {showProgress && (
            <div className="execution-bar">
              <div className="execution-bar-fill" style={{ width: `${state.execution_progress * 100}%` }} />
            </div>
          )}

          {/* Scene legend */}
          <div className="scene-legend">
            <h4>Map Legend</h4>
            <div className="scene-legend-items">
              <div className="scene-legend-item"><span className="scene-legend-color" style={{ background: '#00E5FF' }} /> Start</div>
              <div className="scene-legend-item"><span className="scene-legend-color" style={{ background: '#FFB300' }} /> Goal</div>
              <div className="scene-legend-item"><span className="scene-legend-color" style={{ background: '#00FF66' }} /> Safe Path</div>
              <div className="scene-legend-item"><span className="scene-legend-color" style={{ background: '#FF5555' }} /> Unsafe Path</div>
              <div className="scene-legend-item"><span className="scene-legend-color" style={{ background: '#FF3366' }} /> Obstacles</div>
            </div>
          </div>
        </div>

        <StatusPanel state={state} />
      </div>

      {/* Bottom Metrics */}
      <MetricsBar state={state} />
    </div>
  )
}

function AppWithErrorBoundary() {
  return (
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  )
}

export default AppWithErrorBoundary
