import ImageSlider from './ImageSlider'

const comparisons = [
  {
    id: 'scene-0',
    title: 'Scene 1 — Fast Camera Pan',
    gt: '/images/comparisons/gt_0.png',
    baseline: '/images/comparisons/baseline_0.png',
    best: '/images/comparisons/best_0.png',
  },
  {
    id: 'scene-1',
    title: 'Scene 2 — Object Motion',
    gt: '/images/comparisons/gt_1.png',
    baseline: '/images/comparisons/baseline_1.png',
    best: '/images/comparisons/best_1.png',
  },
  {
    id: 'scene-4',
    title: 'Scene 3 — Complex Motion',
    gt: '/images/comparisons/gt_4.png',
    baseline: '/images/comparisons/baseline_4.png',
    best: '/images/comparisons/best_4.png',
  },
  {
    id: 'scene-6',
    title: 'Scene 4 — Fine Details',
    gt: '/images/comparisons/gt_6.png',
    baseline: '/images/comparisons/baseline_6.png',
    best: '/images/comparisons/best_6.png',
  },
]

export default function ComparisonGrid() {
  return (
    <div className="space-y-10">
      {/* GT vs Baseline */}
      <div>
        <h3 className="text-lg font-medium text-white mb-4">
          Ground Truth vs RGB Baseline
          <span className="text-neutral-500 text-sm font-normal ml-2">
            (drag the slider)
          </span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {comparisons.slice(0, 2).map(c => (
            <div key={c.id + '-gt-base'}>
              <p className="text-neutral-400 text-sm mb-2">{c.title}</p>
              <ImageSlider
                beforeSrc={c.gt}
                afterSrc={c.baseline}
                beforeLabel="Ground Truth"
                afterLabel="RGB Baseline"
                afterMetric="29.95 dB"
              />
            </div>
          ))}
        </div>
      </div>

      {/* GT vs Best Model */}
      <div>
        <h3 className="text-lg font-medium text-white mb-4">
          Ground Truth vs EventWarpNetV2
          <span className="text-emerald-400 text-sm font-normal ml-2">
            (+4.14 dB improvement)
          </span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {comparisons.map(c => (
            <div key={c.id + '-gt-best'}>
              <p className="text-neutral-400 text-sm mb-2">{c.title}</p>
              <ImageSlider
                beforeSrc={c.gt}
                afterSrc={c.best}
                beforeLabel="Ground Truth"
                afterLabel="EventWarpNetV2"
                afterMetric="34.09 dB"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Baseline vs Best */}
      <div>
        <h3 className="text-lg font-medium text-white mb-4">
          RGB Baseline vs EventWarpNetV2
          <span className="text-neutral-500 text-sm font-normal ml-2">
            (see the difference events make)
          </span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {comparisons.slice(0, 2).map(c => (
            <div key={c.id + '-base-best'}>
              <p className="text-neutral-400 text-sm mb-2">{c.title}</p>
              <ImageSlider
                beforeSrc={c.baseline}
                afterSrc={c.best}
                beforeLabel="RGB Baseline"
                beforeMetric="29.95 dB"
                afterLabel="EventWarpNetV2"
                afterMetric="34.09 dB"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
