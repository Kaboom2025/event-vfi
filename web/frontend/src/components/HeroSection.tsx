export default function HeroSection() {
  return (
    <section className="relative py-20 px-6 md:px-12 bg-neutral-950">
      <div className="max-w-5xl mx-auto text-center">
        <p className="text-cyan-400 font-medium text-sm tracking-wide uppercase mb-4">
          UW CSE 493 — Neuromorphic Vision
        </p>
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6 tracking-tight">
          Event-Guided Video
          <br />
          Frame Interpolation
        </h1>
        <p className="text-neutral-400 text-lg md:text-xl max-w-2xl mx-auto mb-10">
          Using synthetic event cameras to reconstruct sharp, high-framerate video.
          Our best model achieves{' '}
          <span className="text-emerald-400 font-semibold">34.09 dB PSNR</span> — a{' '}
          <span className="text-emerald-400 font-semibold">+4.14 dB</span> improvement
          over RGB-only baselines.
        </p>

        <div className="max-w-3xl mx-auto rounded-lg overflow-hidden border border-neutral-800 shadow-2xl">
          <img
            src="/images/hero-comparison.gif"
            alt="Animated comparison of frame interpolation across models"
            className="w-full"
          />
        </div>

        <div className="mt-8 flex flex-wrap justify-center gap-3">
          {['Python', 'PyTorch', 'v2e', 'React'].map(tag => (
            <span
              key={tag}
              className="text-xs font-medium px-3 py-1 rounded-full bg-neutral-800 text-neutral-300 border border-neutral-700"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}
