import { useRef, useState, useCallback, useEffect } from 'react'

interface ImageSliderProps {
  beforeSrc: string
  afterSrc: string
  beforeLabel: string
  afterLabel: string
  beforeMetric?: string
  afterMetric?: string
}

export default function ImageSlider({
  beforeSrc,
  afterSrc,
  beforeLabel,
  afterLabel,
  beforeMetric,
  afterMetric,
}: ImageSliderProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [position, setPosition] = useState(50)
  const [isDragging, setIsDragging] = useState(false)

  const updatePosition = useCallback((clientX: number) => {
    const container = containerRef.current
    if (!container) return
    const rect = container.getBoundingClientRect()
    const x = clientX - rect.left
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setPosition(pct)
  }, [])

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    setIsDragging(true)
    updatePosition(e.clientX)
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  }, [updatePosition])

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!isDragging) return
    updatePosition(e.clientX)
  }, [isDragging, updatePosition])

  const handlePointerUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (document.activeElement !== containerRef.current) return
      if (e.key === 'ArrowLeft') setPosition(p => Math.max(0, p - 2))
      if (e.key === 'ArrowRight') setPosition(p => Math.min(100, p + 2))
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  return (
    <div
      ref={containerRef}
      className="slider-container relative rounded-lg overflow-hidden border border-neutral-800 bg-neutral-900"
      tabIndex={0}
      role="slider"
      aria-label={`Compare ${beforeLabel} and ${afterLabel}`}
      aria-valuenow={Math.round(position)}
      aria-valuemin={0}
      aria-valuemax={100}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      {/* After image (full width, bottom layer) */}
      <img src={afterSrc} alt={afterLabel} className="w-full" draggable={false} />

      {/* Before image (clipped) */}
      <div
        className="absolute inset-0"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      >
        <img src={beforeSrc} alt={beforeLabel} className="w-full" draggable={false} />
      </div>

      {/* Divider line */}
      <div
        className="absolute top-0 bottom-0 w-0.5 bg-white/80 z-10"
        style={{ left: `${position}%`, transform: 'translateX(-50%)' }}
      >
        {/* Handle */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white/90 border-2 border-neutral-300 shadow-lg flex items-center justify-center">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M5 3L2 8L5 13" stroke="#333" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M11 3L14 8L11 13" stroke="#333" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </div>

      {/* Labels */}
      <div className="absolute top-3 left-3 z-20">
        <span className="bg-black/70 text-white text-xs font-medium px-2 py-1 rounded">
          {beforeLabel}
          {beforeMetric && <span className="ml-1 text-cyan-400">{beforeMetric}</span>}
        </span>
      </div>
      <div className="absolute top-3 right-3 z-20">
        <span className="bg-black/70 text-white text-xs font-medium px-2 py-1 rounded">
          {afterLabel}
          {afterMetric && <span className="ml-1 text-emerald-400">{afterMetric}</span>}
        </span>
      </div>

      {/* Drag hint */}
      {!isDragging && position === 50 && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20 animate-pulse">
          <span className="bg-black/60 text-white/70 text-xs px-2 py-1 rounded">
            Drag to compare
          </span>
        </div>
      )}
    </div>
  )
}
