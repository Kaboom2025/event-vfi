import { type ReactNode } from 'react'

interface SectionProps {
  id?: string
  title: string
  subtitle?: string
  children: ReactNode
  dark?: boolean
}

export default function Section({ id, title, subtitle, children, dark }: SectionProps) {
  return (
    <section
      id={id}
      className={`py-16 px-6 md:px-12 ${dark ? 'bg-neutral-950' : 'bg-neutral-900/50'}`}
    >
      <div className="max-w-5xl mx-auto">
        <h2 className="text-2xl md:text-3xl font-semibold text-white mb-2">{title}</h2>
        {subtitle && (
          <p className="text-neutral-400 mb-8 max-w-2xl">{subtitle}</p>
        )}
        {!subtitle && <div className="mb-8" />}
        {children}
      </div>
    </section>
  )
}
