const results = [
  { model: 'RGB Baseline', type: 'baseline', params: '17.3M', psnr: 29.95, ssim: 0.8708, best: false },
  { model: 'DualEncoder (zero events)', type: 'ablation', params: '27.9M', psnr: 28.0, ssim: 0.8296, best: false },
  { model: 'DualEncoder', type: 'dual', params: '27.9M', psnr: 31.0, ssim: 0.886, best: false },
  { model: 'DualEncoder + Perceptual', type: 'dual', params: '27.9M', psnr: 31.24, ssim: 0.891, best: false },
  { model: 'EventWarpNet v1', type: 'warp', params: '13.4M', psnr: 32.73, ssim: 0.9219, best: false },
  { model: 'EventWarpNetV2', type: 'warp_v2', params: '15.1M', psnr: 33.83, ssim: 0.9408, best: false },
  { model: 'EventWarpNetV2 + Perceptual', type: 'warp_v2', params: '15.1M', psnr: 34.09, ssim: 0.945, best: true },
]

export default function ResultsTable() {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead>
          <tr className="border-b border-neutral-700 text-neutral-400">
            <th className="py-3 px-4 font-medium">Model</th>
            <th className="py-3 px-4 font-medium text-right">Params</th>
            <th className="py-3 px-4 font-medium text-right">PSNR (dB)</th>
            <th className="py-3 px-4 font-medium text-right">SSIM</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr
              key={r.model}
              className={`border-b border-neutral-800 ${
                r.best
                  ? 'bg-emerald-500/10 text-white'
                  : r.type === 'ablation'
                    ? 'text-neutral-500'
                    : 'text-neutral-300'
              }`}
            >
              <td className="py-3 px-4">
                {r.best && <span className="text-emerald-400 mr-1">*</span>}
                {r.model}
                {r.type === 'ablation' && (
                  <span className="ml-2 text-xs text-neutral-600">(control)</span>
                )}
              </td>
              <td className="py-3 px-4 text-right font-mono text-xs">{r.params}</td>
              <td className={`py-3 px-4 text-right font-mono ${r.best ? 'text-emerald-400 font-semibold' : ''}`}>
                {r.psnr.toFixed(2)}
              </td>
              <td className={`py-3 px-4 text-right font-mono ${r.best ? 'text-emerald-400 font-semibold' : ''}`}>
                {r.ssim.toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
