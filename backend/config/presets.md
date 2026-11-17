# Abliterator presets

The styles offered on the Engine panel. Edit the YAML block and reload the page. Each preset is a style prompt, a group for the menu, sampler settings and seeds; `x` and `y` name the directions the XY pad steers. Seeds marked `heard: false` have not yet been listened to and judged.

The knobs' musical ranges (Tune) were measured on the four ambient presets, and the brightness and pulse directions were made from ambient prompts with no drums. On the other presets the knobs keep those ranges until a Tune pass has been run on them.

Prompts typed on the Engine panel are kept in `data/prompts.json` and listed under Your prompts; they are not part of this file.

`cfg`, `temperature` and `top_k` here are MusicGen's settings. Magenta RT 2 starts from its own (guidance 3, temperature 1.3, top-k 40) unless a preset gives an `mrt2` block.

```yaml
presets:
  - id: warm-drone
    group: Ambient
    name: Warm drone
    prompt: slow ambient drone, warm analogue pads, no drums
    cfg: 1.0
    temperature: 1.0
    top_k: 250
    seeds: [1, 2, 3]
    heard: false
    x: brightness
    y: pulse
  - id: shimmer
    group: Ambient
    name: Shimmering pad
    prompt: evolving ambient pad texture, shimmering, reverb, calm, no percussion
    cfg: 1.0
    temperature: 1.0
    top_k: 250
    seeds: [2, 5, 8]
    heard: false
    x: brightness
    y: pulse
  - id: dark-field
    group: Ambient
    name: Dark field
    prompt: deep low drone, dark atmospheric soundscape, field recording textures
    cfg: 1.0
    temperature: 1.0
    top_k: 250
    seeds: [3, 6, 9]
    heard: false
    x: brightness
    y: pulse
  - id: glass
    group: Ambient
    name: Glass chords
    prompt: soft glassy ambient chords, slow attack, no beat
    cfg: 1.0
    temperature: 1.0
    top_k: 250
    seeds: [4, 7, 10]
    heard: false
    x: brightness
    y: pulse
  - {id: deep-techno, group: Electronic, name: Deep techno, prompt: "deep minimal techno, hypnotic kick drum, rolling bassline, dark synth stabs", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: synthwave, group: Electronic, name: Synthwave, prompt: "retro synthwave, analogue arpeggios, gated drums, 1980s", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: dnb, group: Electronic, name: Liquid drum and bass, prompt: "liquid drum and bass, fast breakbeats, deep sub bass, atmospheric pads", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: idm, group: Electronic, name: Glitch IDM, prompt: "glitchy IDM, granular textures, stuttering percussion, detuned synths", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: lofi, group: Beats, name: Lo-fi hip hop, prompt: "lo-fi hip hop beat, dusty drums, mellow jazz piano, vinyl crackle", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: dub, group: Beats, name: Roots dub, prompt: "roots dub reggae, deep bass, echoing snare, skanking guitar", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: afrobeat, group: Beats, name: Afrobeat, prompt: "afrobeat groove, horn section, funky guitar, polyrhythmic percussion", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: piano, group: Acoustic, name: Solo piano, prompt: "solo felt piano, slow, intimate, sparse", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: folk, group: Acoustic, name: Folk guitar, prompt: "fingerpicked acoustic guitar, warm folk, gentle", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: nylon, group: Acoustic, name: Nylon guitar, prompt: "nylon-string classical guitar, slow fingerpicked arpeggios, warm and intimate", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: open-tuning, group: Acoustic, name: Open-tuned guitar, prompt: "steel-string acoustic guitar in open tuning, ringing drone strings, slow and spacious", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: mandolin, group: Acoustic, name: Guitar and mandolin, prompt: "acoustic guitar and mandolin duet, warm folk, gentle", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: old-time, group: Acoustic, name: Banjo and fiddle, prompt: "old-time clawhammer banjo and fiddle, gentle, rustic", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: harp, group: Acoustic, name: Celtic harp, prompt: "celtic harp, gentle plucked melody, airy and warm", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: kora, group: Acoustic, name: Kora, prompt: "West African kora, flowing plucked patterns, calm and warm", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: jazz-trio, group: Acoustic, name: Jazz trio, prompt: "jazz piano trio, brushed drums, walking double bass", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: quartet, group: Orchestral, name: String quartet, prompt: "string quartet, slow and romantic, cello and violins", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: cinematic, group: Orchestral, name: Cinematic swell, prompt: "cinematic orchestral swell, strings and brass, timpani, epic", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: gamelan, group: Orchestral, name: Gamelan, prompt: "Balinese gamelan, metallic percussion, interlocking patterns", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: concrete, group: Experimental, name: Musique concrète, prompt: "musique concrete, tape loops, found sounds, reversed textures", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
  - {id: noise, group: Experimental, name: Noise wall, prompt: "harsh noise, distorted feedback, dense static", cfg: 1.0, temperature: 1.0, top_k: 250, seeds: [1, 2, 3], heard: false, x: brightness, y: pulse}
```
