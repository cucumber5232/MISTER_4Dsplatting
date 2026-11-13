# Licenze e uso aziendale

Vincolo del progetto: deve poter essere usato in azienda, quindi **nessuna
base con licenza a pagamento o limitata alla ricerca**. Questo documento dice
che cosa c'è dentro, che cosa è stato escluso e perché, e che cosa resta da
far verificare.

> Non è un parere legale. Le licenze elencate sono state lette il 12/11/2026
> sui file di licenza e sui metadati dei pacchetti; vanno ricontrollate quando
> si aggiorna una dipendenza e prima di una consegna a terzi.

## 1. Che cosa contiene il repository

Tutto il codice in `src/mister4d/` è stato **scritto da zero per questo
progetto**: ingest, sincronizzazione, calibrazione, maschere, visual hull,
modello dinamico (Gaussiane locali nel tempo), training,
valutazione, export, player WebGL, interfaccia guidata (MISTER 4D Studio) e
generatore di dati sintetici. Non è stato
copiato codice da repository di ricerca.

Dal mondo Gaussian Splatting si usa **solo gsplat** (Apache-2.0): il
rasterizzatore CUDA e la strategia di densificazione.

## 2. Dipendenze

| Componente | Licenza | Uso | Note |
|---|---|---|---|
| [gsplat](https://github.com/nerfstudio-project/gsplat) 1.5 | Apache-2.0 | rasterizzazione GPU, densificazione | Non usare le funzioni `rasterization_inria_wrapper` / `rasterization_2dgs_inria_wrapper`: richiamano il rasterizzatore Inria (vedi §3). Questo progetto non le usa. |
| PyTorch | BSD-3-Clause (più componenti Apache-2.0/BSD) | ottimizzazione | La build CUDA installa le librerie NVIDIA (vedi sotto). |
| NumPy | BSD-3-Clause | calcolo | |
| SciPy | BSD-3-Clause | bundle adjustment, k-d tree, audio | |
| OpenCV (`opencv-python-headless`) | Apache-2.0 | ChArUco, undistort, immagini | La wheel include FFmpeg in versione LGPL. |
| PyYAML | MIT | configurazione | |
| jaxtyping, ninja, rich | MIT / Apache-2.0 / MIT | dipendenze di gsplat | |
| pytest | MIT | solo sviluppo | |

Strumenti esterni, non inclusi nel repository:

- **FFmpeg / ffprobe**: richiamati come programmi esterni (nessun link al
  codice). La licenza è LGPL o GPL secondo come è compilato l'eseguibile
  installato: non incide sul nostro codice, ma conta se si **ridistribuisce**
  FFmpeg insieme a un prodotto.
- **Librerie NVIDIA CUDA** (pacchetti `nvidia-*` installati da PyTorch, driver,
  CUDA Toolkit): software proprietario gratuito, regolato dall'EULA NVIDIA.
  L'uso interno è normale prassi; l'EULA va letto se si ridistribuisce.

Il player web (`src/mister4d/viewer/`) e la pagina di MISTER 4D Studio
(`src/mister4d/studio/`) non hanno dipendenze: nessuna libreria JavaScript,
nessun font e nessuna icona di terzi (i caratteri sono quelli del sistema, i
disegni sono SVG scritti a mano). Il server di Studio usa solo la libreria
standard di Python.

## 3. Che cosa è stato escluso, e perché

Il punto critico segnalato nella guida è reale: molti repository hanno
un'etichetta MIT/Apache ma **dipendono da codice Inria**, la cui licenza
consente solo ricerca e valutazione non commerciale.

| Progetto | Etichetta del repository | Problema riscontrato |
|---|---|---|
| `graphdeco-inria/gaussian-splatting`, `diff-gaussian-rasterization`, `simple-knn` | "Gaussian-Splatting License" (Inria / MPII) | Uso consentito solo per ricerca non commerciale; l'uso commerciale richiede un accordo con i titolari. |
| `hustvl/4DGaussians` (rif. [4] della guida) | Apache-2.0 | I sottomoduli sono `simple-knn` (Inria), un rasterizzatore derivato da quello Inria e i viewer SIBR: valgono i vincoli Inria. |
| `JonathonLuiten/Dynamic3DGaussians` | MIT | Il file di licenza dichiara che il codice richiede rasterizzatore e parti di codice Inria, con la relativa licenza restrittiva. |
| `yangzf-1023/4C4D` (rif. [5] della guida) | MIT | L'installazione richiede componenti esterni (es. MAtCha) le cui licenze **non sono state verificate**: da controllare una per una prima di qualunque uso. |

Regola pratica: qualunque progetto che installi `diff-gaussian-rasterization`
o `simple-knn` è fuori, anche se il suo file LICENSE dice altro. I metodi
descritti negli articoli si possono reimplementare (è ciò che fa questo
repository su gsplat); il loro codice no.

Altri elementi tipici delle pipeline "umane" da **non** introdurre senza
verifica:

- **Modelli corporei parametrici (SMPL, SMPL-X e derivati)**: gratuiti solo
  per ricerca, l'uso commerciale richiede una licenza a pagamento. Questo
  progetto non usa alcun modello corporeo.
- **Pesi di reti per segmentazione / matting / stima di profondità**: la
  licenza dei pesi è spesso diversa da quella del codice. Qui le maschere si
  ottengono per confronto con il set vuoto, senza reti. Se si aggiunge un
  modello, registrare in questa pagina la licenza del codice **e** dei pesi.
- **Dataset pubblici**: molti dataset multicamera sono rilasciati per sola
  ricerca. Per le prove interne il repository genera dati sintetici propri
  (`mister4d demo`); per qualunque altro dataset leggere prima la licenza.

## 4. Che cosa resta da decidere o verificare

1. **Licenza di questo repository.** Non è stato aggiunto alcun file LICENSE:
   in sua assenza valgono i diritti d'autore ordinari (nessun diritto concesso
   a terzi). Il repository GitHub è però **pubblico**: il codice è leggibile
   da chiunque. Decidere se renderlo privato o scegliere una licenza.
2. **Brevetti.** Una licenza software copre il codice, non eventuali brevetti
   sui metodi. Prima di offrire a clienti un prodotto o servizio basato su
   Gaussian Splatting è opportuna una verifica con un consulente.
3. **Codec.** I video in ingresso e in uscita usano i codec disponibili in
   FFmpeg (H.264 se presente). La distribuzione commerciale di contenuti o
   software con codec brevettati ha regole proprie.
4. **Diritti sulle riprese.** Le registrazioni di persone richiedono
   liberatorie e un trattamento conforme al GDPR: fuori dallo scopo del
   software, ma parte del progetto.

## 5. Regole per chi sviluppa

- Niente copia-incolla da repository di ricerca: si rilegge l'articolo e si
  riscrive.
- Ogni nuova dipendenza si aggiunge alla tabella del §2 con la licenza letta
  sul file LICENSE del pacchetto, non sull'etichetta di GitHub, controllando
  anche sottomoduli e dipendenze indirette.
- Verifica rapida dell'ambiente installato:

```bash
pip install pip-licenses
```

```bash
pip-licenses --from=mixed --order=license
```
