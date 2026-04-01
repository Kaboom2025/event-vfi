import ImageSlider from './ImageSlider'

const scenes = [
  { id: 0, title: 'Scene 1 — Fast Camera Pan' },
  { id: 1, title: 'Scene 2 — Object Motion' },
  { id: 4, title: 'Scene 3 — Complex Motion' },
  { id: 6, title: 'Scene 4 — Fine Details' },
  { id: 2, title: 'Scene 5 — Subtle Motion' },
  { id: 3, title: 'Scene 6 — Texture Detail' },
  { id: 5, title: 'Scene 7 — Low Contrast' },
  { id: 7, title: 'Scene 8 — High Motion' },
]

export default function ComparisonGrid() {
  return (
    <div className="space-y-14">
      {/* Animated GIF comparisons */}
      <div>
        <h3 className="text-lg font-medium text-white mb-2">
          Animated Comparisons
        </h3>
        <p className="text-neutral-400 text-sm mb-6">
          Each GIF cycles through: <span className="text-white">Ground Truth</span> →{' '}
          <span className="text-red-400">RGB Baseline (Ablated)</span> →{' '}
          <span className="text-emerald-400">EventWarpNetV2 (Ours)</span>
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {scenes.map(s => (
            <div key={`gif-${s.id}`}>
              <p className="text-neutral-400 text-xs mb-1.5">{s.title}</p>
              <div className="rounded-lg overflow-hidden border border-neutral-800">
                <img
                  src={`/images/gifs/comparison_${s.id}.gif`}
                  alt={`${s.title}: cycling GT, Baseline, EventWarpNetV2`}
                  className="w-full"
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Interactive sliders: GT vs Best */}
      <div>
        <h3 className="text-lg font-medium text-white mb-2">
          Interactive Sliders
          <span className="text-emerald-400 text-sm font-normal ml-2">
            Ground Truth vs EventWarpNetV2
          </span>
        </h3>
        <p className="text-neutral-500 text-sm mb-6">Drag to compare</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {scenes.slice(0, 4).map(s => (
            <div key={`slider-gt-best-${s.id}`}>
              <p className="text-neutral-400 text-sm mb-2">{s.title}</p>
              <ImageSlider
                beforeSrc={`/images/comparisons/gt_${s.id}.png`}
                afterSrc={`/images/comparisons/best_${s.id}.png`}
                beforeLabel="Ground Truth"
                afterLabel="EventWarpNetV2 (Ours)"
                afterMetric="34.09 dB"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Interactive sliders: Baseline vs Best */}
      <div>
        <h3 className="text-lg font-medium text-white mb-2">
          Ablated vs Ours
          <span className="text-neutral-500 text-sm font-normal ml-2">
            See the difference events make
          </span>
        </h3>
        <p className="text-neutral-500 text-sm mb-6">Drag to compare</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {scenes.slice(0, 4).map(s => (
            <div key={`slider-base-best-${s.id}`}>
              <p className="text-neutral-400 text-sm mb-2">{s.title}</p>
              <ImageSlider
                beforeSrc={`/images/comparisons/baseline_${s.id}.png`}
                afterSrc={`/images/comparisons/best_${s.id}.png`}
                beforeLabel="RGB Baseline"
                beforeMetric="29.95 dB"
                afterLabel="EventWarpNetV2 (Ours)"
                afterMetric="34.09 dB"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
