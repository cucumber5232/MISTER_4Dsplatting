# MISTER 4D in Unity, per Quest 3

Un componente che riproduce in Unity una scena esportata da MISTER 4D, pensato
per girare su un visore autonomo.

> **Stato: non ancora aperto in Unity né provato su un visore.** È stato
> scritto senza Unity a disposizione. Qui sotto c'è che cosa è stato
> verificato lo stesso e che cosa resta da verificare alla prima apertura.

## Che cosa c'è

```
unity/Mister4D/
  Mister4DPlayer.cs               il componente da aggiungere a un oggetto
  Mister4DChunk.cs                lettura dei dati, scelta e ordinamento delle Gaussiane (non usa Unity)
  Resources/MisterSplat4D.shader  il disegno
```

Nessuna dipendenza: né pacchetti, né codice di terzi.

## Come si usa

1. Unity 2022.3 LTS o successivo, progetto 3D (Built-in oppure URP).
2. Copiare la cartella `unity/Mister4D` dentro `Assets/` del progetto.
3. Preparare la scena con MISTER 4D. Da Studio: nel passo Risultato,
   «Versione leggera per visore autonomo»; la cartella da usare è
   `risultato/scena_visore`. Da terminale:

   ```bash
   mister4d lighten runs/take01 --out runs/take01_visore --budget 80000
   ```

   ```bash
   mister4d export runs/take01_visore --format engine --out scena_visore
   ```

4. Copiare la cartella ottenuta in `Assets/StreamingAssets/` (creare
   `StreamingAssets` se non c'è).
5. In una scena: oggetto vuoto → *Add Component* → *MISTER 4D* → *Player*.
   Nel campo **Folder** scrivere il nome della cartella (`scena_visore`).
6. Premere Play.

L'oggetto è il soggetto: si sposta, si ruota e si scala come qualunque altro.
Le misure sono quelle vere, in metri. L'origine è quella della calibrazione:
il centro del set, oppure il pannello appoggiato a terra se è stato indicato
(solo in quel caso i piedi del soggetto stanno a quota zero).

Da script: `Play()`, `Pause()`, `Seek(frame)`; `Frame`, `FrameCount`,
`IsReady`; `VisibleNow` dice quante Gaussiane sono state disegnate nell'ultimo
fotogramma; `Error` dice che cosa non ha funzionato.

### Impostazioni per Quest 3

- *Build Settings*: Android. *XR Plug-in Management*: OpenXR con le
  funzionalità Meta Quest.
- Grafica: Vulkan oppure OpenGL ES 3. Rendering stereo: Multiview.
- Spazio colore: Linear o Gamma, lo shader ne tiene conto.
- Quante Gaussiane: il numero che conta è quante sono visibili insieme
  (`--budget`). Chi ha misurato altri lettori Unity su Quest 3 autonomo riporta
  72 fotogrammi al secondo con 50–100 mila Gaussiane e circa 59 con 200 mila
  ([SplatXR](https://radiancefields.com/splatxr-brings-interactive-gaussian-splats-to-quest-3)):
  80 mila è un punto di partenza, da correggere guardando `VisibleNow` e i
  fotogrammi al secondo sul visore.

## Che cosa è stato verificato, e come

Con `pytest tests/test_unity.py` dalla cartella principale:

- la cartella esportata per i motori grafici contiene gli stessi byte del
  player web, già collaudato in un browser;
- `Mister4DChunk.cs` è stato compilato ed eseguito fuori da Unity (con mono)
  su scene esportate davvero: per ogni istante sceglie le stesse Gaussiane che
  l'esportatore dichiara visibili, e le ordina da dietro in avanti;
- i conti dello shader (posizione sullo schermo e forma di ogni Gaussiana,
  scambio degli assi tra il sistema delle riprese e quello di Unity) sono stati
  trascritti riga per riga e confrontati con la camera usata dal resto del
  programma: coincidono.

Inoltre `Mister4DPlayer.cs` è stato controllato dal compilatore contro una
copia ridotta, scritta a mano, delle funzioni di Unity che usa: dice che il
codice è coerente con sé stesso, non che quelle funzioni si comportino come
previsto.

## Che cosa resta da verificare

Alla prima apertura, in quest'ordine. Per ogni punto che non torna serve il
testo dell'errore oppure una foto di quello che si vede.

1. **Compila?** Nessun errore nella Console di Unity, né dagli script né dallo
   shader.
2. **Si vede il soggetto?** In piedi, non sdraiato né specchiato, e delle
   dimensioni giuste.
3. **L'ordine è giusto?** Girandogli intorno, le parti dietro non devono
   apparire davanti.
4. **Stereo.** Sul visore le due lenti devono mostrare la stessa scena, con la
   giusta profondità.
5. **Colori.** Né slavati né troppo scuri rispetto al player web.
6. **Fluidità.** Fotogrammi al secondo sul visore, insieme a `VisibleNow`.
7. **Passaggi tra spezzoni.** Nelle riprese lunghe, nessuno scatto ogni due
   secondi.

## Limiti noti

- Il colore non cambia con il punto di vista.
- Scelta e ordinamento delle Gaussiane avvengono sul processore, nel thread
  principale, a ogni fotogramma.
- La scena non riceve luci né ombre: ha la luce delle riprese.
- Rispetto ad altri oggetti trasparenti l'ordine è deciso oggetto per oggetto.

## Il formato, per chi vuole leggerlo altrove (Unreal)

Per Unreal non c'è ancora un lettore. Il formato è lo stesso ed è volutamente
semplice.

`scene.json` contiene solo numeri, testi ed elenchi di numeri: fotogrammi,
fotogrammi al secondo, soglia di opacità (`min_opacity`) e, per ogni blocco,
il file e gli intervalli per riportare gli interi alle grandezze vere. Le
coordinate sono in metri, sistema destrorso, asse z verso l'alto.

Ogni blocco (`chunks/0000.bin`, compresso con gzip) copre una parte della
ripresa, i fotogrammi da `first` a `last` escluso. Contiene `count` Gaussiane,
23 byte ciascuna, in cinque parti una dopo l'altra:

| Parte | Tipo | Contenuto |
|---|---|---|
| A | 4 × uint16 | posizione x y z, istante |
| B | 4 × uint8 | velocità x y z, durata |
| C | 4 × uint8 | rotazione (quaternione w x y z) |
| D | 4 × uint8 | logaritmo delle scale x y z, opacità di picco |
| E | 3 × uint8 | colore r g b |

Con `a`, `b`, `c`, `d` i valori di una Gaussiana e `t` l'istante da mostrare,
in fotogrammi:

```
istante    = time_start + a.w / 65535 * time_span
dt         = t - istante
durata     = exp(life_log_min + (life_log_max - life_log_min) * b.w / 254)     b.w = 255: sempre visibile
codice     = (b.xyz - 128) / 127
posizione  = bbox_min + a.xyz / 65535 * (bbox_max - bbox_min) + codice * |codice| * vel_max * dt
opacità    = d.w / 255 * exp(-0.5 * (dt / durata)^2)                           sotto min_opacity non si disegna
scala      = exp(scale_log_min + (scale_log_max - scale_log_min) * d.xyz / 255)
rotazione  = normalizza(c / 255 * 2 - 1)
```

Le prime `long_lived` Gaussiane del blocco vivono a lungo e vanno sempre
considerate; le altre sono ordinate per istante, e basta guardare quelle
entro `reach` fotogrammi da `t`. Le Gaussiane si disegnano da dietro in
avanti, con colori premoltiplicati. `Mister4DChunk.cs` e lo shader sono il
riferimento.
