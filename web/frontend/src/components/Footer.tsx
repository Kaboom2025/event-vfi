export default function Footer() {
  return (
    <footer className="py-12 px-6 border-t border-neutral-800 bg-neutral-950">
      <div className="max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="text-neutral-500 text-sm">
          UW CSE 493 — Neuromorphic Vision, 2025
        </div>
        <div className="flex gap-6">
          <a
            href="https://github.com/Kaboom2025/event-vfi"
            target="_blank"
            rel="noopener noreferrer"
            className="text-neutral-400 hover:text-white transition-colors text-sm"
          >
            GitHub
          </a>
          <a
            href="https://saalikahmed.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-neutral-400 hover:text-white transition-colors text-sm"
          >
            Portfolio
          </a>
        </div>
      </div>
    </footer>
  )
}
