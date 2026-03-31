import HeroSection from './components/HeroSection'
import Section from './components/Section'
import ComparisonGrid from './components/ComparisonGrid'
import ResultsTable from './components/ResultsTable'
import Footer from './components/Footer'

export default function App() {
  return (
    <div className="min-h-screen bg-neutral-950">
      <HeroSection />

      {/* The Problem */}
      <Section
        id="problem"
        title="The Problem"
        subtitle="Standard cameras are blind between frames."
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          <div className="text-neutral-300 space-y-4">
            <p>
              A 30 fps camera captures just 30 snapshots per second. Between those frames,
              the world keeps moving — but the camera sees nothing. When you try to
              reconstruct what happened in between (frame interpolation), fast-moving objects
              blur, ghost, or disappear entirely.
            </p>
            <p>
              <span className="text-cyan-400 font-medium">Event cameras</span> solve this by
              firing at every pixel that changes brightness, with microsecond resolution. They
              capture the motion that traditional cameras miss.
            </p>
            <p>
              We use <span className="text-cyan-400 font-medium">v2e</span> to simulate event
              camera data from standard video, then fuse these events with RGB frames to guide
              interpolation — especially in high-motion regions.
            </p>
          </div>
          <div className="rounded-lg overflow-hidden border border-neutral-800">
            <img
              src="/images/events-encoding.webp"
              alt="What event cameras encode: brightness changes at microsecond resolution"
              className="w-full"
              loading="lazy"
            />
          </div>
        </div>
      </Section>

      {/* Interactive Comparisons */}
      <Section
        id="comparisons"
        title="Interactive Comparisons"
        subtitle="Drag the sliders to compare frame interpolation quality across models."
        dark
      >
        <ComparisonGrid />
      </Section>

      {/* How It Works */}
      <Section
        id="architecture"
        title="How It Works"
        subtitle="A two-stage flow estimation pipeline with learned attention gates."
      >
        <div className="space-y-8">
          <div className="rounded-lg overflow-hidden border border-neutral-800">
            <img
              src="/images/architecture.webp"
              alt="Architecture comparison of all four models"
              className="w-full"
              loading="lazy"
            />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-neutral-300">
            <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
              <h4 className="text-white font-medium mb-2">Stage 1: Event Flow</h4>
              <p>
                Events encode precise motion between frames. A FlowNet takes the 5-bin event
                voxel and estimates coarse bidirectional optical flow.
              </p>
            </div>
            <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
              <h4 className="text-white font-medium mb-2">Stage 2: RGB Refinement</h4>
              <p>
                The coarse flow is refined using RGB appearance — a second network predicts
                a delta flow that corrects texture-level details events can't resolve.
              </p>
            </div>
            <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
              <h4 className="text-white font-medium mb-2">Attention Gates</h4>
              <p>
                Learned attention gates focus refinement on moving regions — where events fire
                most — with zero explicit supervision.
              </p>
            </div>
          </div>
        </div>
      </Section>

      {/* Results */}
      <Section
        id="results"
        title="Results"
        subtitle="Seven model variants evaluated on the Vimeo-90k triplet test set (3,782 clips)."
        dark
      >
        <div className="space-y-8">
          <div className="bg-neutral-900 rounded-lg border border-neutral-800 overflow-hidden">
            <ResultsTable />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
              <div className="text-3xl font-bold text-emerald-400">+3.0 dB</div>
              <div className="text-neutral-400 text-sm mt-1">Events vs no events</div>
              <div className="text-neutral-500 text-xs mt-1">DualEncoder vs zero-events control</div>
            </div>
            <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
              <div className="text-3xl font-bold text-emerald-400">+2.85 dB</div>
              <div className="text-neutral-400 text-sm mt-1">Flow vs direct synthesis</div>
              <div className="text-neutral-500 text-xs mt-1">EventWarpNetV2 vs DualEncoder</div>
            </div>
            <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
              <div className="text-3xl font-bold text-emerald-400">+4.14 dB</div>
              <div className="text-neutral-400 text-sm mt-1">Full pipeline vs RGB only</div>
              <div className="text-neutral-500 text-xs mt-1">EventWarpNetV2+Perc vs Baseline</div>
            </div>
          </div>

          <div className="rounded-lg overflow-hidden border border-neutral-800">
            <img
              src="/images/ablation.webp"
              alt="Ablation bar chart comparing PSNR and SSIM across models"
              className="w-full"
              loading="lazy"
            />
          </div>
        </div>
      </Section>

      {/* What the Network Learns */}
      <Section
        id="visualizations"
        title="What the Network Learns"
        subtitle="The network discovers where to focus — without being told."
      >
        <div className="space-y-8">
          <div>
            <p className="text-neutral-400 text-sm mb-3">
              Attention gates learn to focus refinement on high-motion regions where events
              fire most. No explicit supervision — the network discovers this from the
              reconstruction loss alone.
            </p>
            <div className="rounded-lg overflow-hidden border border-neutral-800">
              <img
                src="/images/attention-gates.webp"
                alt="Learned attention gates highlighting motion regions"
                className="w-full"
                loading="lazy"
              />
            </div>
          </div>

          <div>
            <p className="text-neutral-400 text-sm mb-3">
              Two-stage flow decomposition: events produce coarse motion estimates, then
              RGB refines texture-level details. The combined flow captures both fast motion
              and fine appearance.
            </p>
            <div className="rounded-lg overflow-hidden border border-neutral-800">
              <img
                src="/images/flow-viz.webp"
                alt="Two-stage flow visualization: coarse event flow + refined RGB flow"
                className="w-full"
                loading="lazy"
              />
            </div>
          </div>
        </div>
      </Section>

      {/* Events Help Where Needed Most */}
      <Section
        id="motion"
        title="Events Help Where Needed Most"
        subtitle="The benefit of event data concentrates on high-motion scenes."
        dark
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          <div className="rounded-lg overflow-hidden border border-neutral-800">
            <img
              src="/images/motion-scatter.webp"
              alt="Motion-stratified scatter plot showing PSNR gain vs event density"
              className="w-full"
              loading="lazy"
            />
          </div>
          <div className="text-neutral-300 space-y-4">
            <p>
              When motion is minimal, RGB-only interpolation works fine. But as motion
              increases, RGB baselines degrade rapidly while event-guided models maintain
              quality.
            </p>
            <p>
              The scatter plot shows each test clip's PSNR gain (EventWarpNetV2 minus
              Baseline) versus its event density — a proxy for motion magnitude. The
              trend is clear: <span className="text-emerald-400 font-medium">events
              help most where they're needed most</span>.
            </p>
          </div>
        </div>
      </Section>

      <Footer />
    </div>
  )
}
