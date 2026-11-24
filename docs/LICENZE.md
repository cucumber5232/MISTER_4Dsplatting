# Licenze

Il progetto deve poter essere usato in azienda, quindi non contiene componenti
con licenza a pagamento o di sola ricerca. Qui c'è l'elenco di quello che usa,
di quello che è stato escluso e di quello che resta da controllare.

Questo documento non è un parere legale. Le licenze sono state lette il
12/11/2026 sui file di licenza e sui metadati dei pacchetti. Vanno
ricontrollate quando si aggiorna una dipendenza e prima di una consegna a
terzi.

## 1. Il codice di questo repository

Il codice è di proprietà di MISTER Smart Innovation, tutti i diritti
riservati. Il testo è nel file [LICENSE](../LICENSE). Il repository su GitHub
è pubblico, quindi il codice si può leggere, ma per usarlo serve
un'autorizzazione scritta.

Tutto quello che sta in `src/mister4d/` è stato scritto per questo progetto:
ingest, sincronizzazione, calibrazione, maschere, visual hull, modello
dinamico, training, valutazione, export, player WebGL, MISTER 4D Studio e
generatore di dati sintetici. Non è stato copiato codice da repository di
ricerca.

Dal mondo Gaussian Splatting si usa solo gsplat (Apache-2.0), per il
rasterizzatore CUDA e la strategia di densificazione.

Alcune scelte di metodo vengono da articoli scientifici pubblicati, elencati
in [PIPELINE.md](PIPELINE.md) alla voce «Idee recenti». Di quegli articoli è
stata usata l'idea, riscritta qui. Il codice dei loro autori non è stato
usato, perché spesso ha licenze di sola ricerca o dipende da codice che le
ha. Un'idea descritta in un articolo non è coperta dal diritto d'autore, ma
può essere coperta da un brevetto (vedi §4, punto 1).

Lo stesso vale per Open4DGS (novembre 2026). La parte di preparazione,
Open4DGS-Core, ha licenza MIT: il suo codice si potrebbe riusare tenendo la
nota di copyright, ma finora non è stato copiato. L'addestratore,
Open4DGS-Trigs, ha licenza AGPL-3.0: se ne possono prendere solo le idee, e
il suo codice non va copiato né adattato. I modelli che Open4DGS usa per i
punti di partenza non vengono usati qui: i pesi di RoMa v2 non hanno una
licenza dichiarata e contengono DINOv3, che ha una licenza propria, e quelli
di MEMFOF sono stati allenati in parte su dati per uso non commerciale. Le
riprese di esempio di Open4DGS sono solo per prove private non commerciali:
niente di ricavato da quelle può essere mostrato o consegnato.

## 2. Dipendenze

| Componente | Licenza | Uso | Note |
|---|---|---|---|
| [gsplat](https://github.com/nerfstudio-project/gsplat) 1.5 | Apache-2.0 | rasterizzazione GPU, densificazione | Le funzioni `rasterization_inria_wrapper` e `rasterization_2dgs_inria_wrapper` richiamano il rasterizzatore Inria (vedi §3) e non vanno usate. Questo progetto non le usa. |
| PyTorch | BSD-3-Clause (più componenti Apache-2.0/BSD) | ottimizzazione | La build CUDA installa le librerie NVIDIA (vedi sotto). |
| NumPy | BSD-3-Clause | calcolo | |
| SciPy | BSD-3-Clause | bundle adjustment, k-d tree, audio | |
| OpenCV (`opencv-python-headless`) | Apache-2.0 | ChArUco, undistort, immagini | La wheel include FFmpeg in versione LGPL. |
| PyYAML | MIT | configurazione | |
| jaxtyping, ninja, rich | MIT / Apache-2.0 / MIT | dipendenze di gsplat | |
| pytest | MIT | solo sviluppo | |

Strumenti esterni, non inclusi nel repository:

- FFmpeg e ffprobe sono richiamati come programmi esterni. La loro licenza è
  LGPL o GPL secondo come è compilato l'eseguibile installato. Non tocca il
  nostro codice, ma conta se FFmpeg viene ridistribuito insieme a un prodotto.
- Le librerie NVIDIA CUDA (pacchetti `nvidia-*` installati da PyTorch, driver,
  CUDA Toolkit) sono software proprietario gratuito, regolato dall'EULA
  NVIDIA. Per l'uso interno non ci sono problemi. Se si ridistribuiscono va
  letto l'EULA.

Il player web (`src/mister4d/viewer/`) e la pagina di MISTER 4D Studio
(`src/mister4d/studio/`) non hanno dipendenze: nessuna libreria JavaScript,
nessun font e nessuna icona di terzi. I caratteri sono quelli del sistema e i
disegni sono SVG scritti a mano. Il server di Studio usa solo la libreria
standard di Python.

### Dati inclusi nel pacchetto

| Dato | Licenza | Uso | Note |
|---|---|---|---|
| `src/mister4d/assets/human.npz`, ricavato da [MakeHuman](https://github.com/makehumancommunity/makehuman) | CC0 1.0 (dati di MakeHuman) | figura umana delle riprese sintetiche (`mister4d demo --subject human`) | Da MakeHuman vengono solo mesh di base, un target di forma, scheletro e pesi, tutti rilasciati CC0 (`LICENSE.md`, sezione C). Il programma MakeHuman è AGPL e non è usato né incluso. Vestiti, capelli e texture sono generati da `scripts/build_human_asset.py`. Dettagli in `src/mister4d/assets/HUMAN.md`. |

## 3. Che cosa è stato escluso

Molti repository hanno un'etichetta MIT o Apache ma dipendono da codice Inria,
la cui licenza consente solo ricerca e valutazione non commerciale.

| Progetto | Etichetta del repository | Problema |
|---|---|---|
| `graphdeco-inria/gaussian-splatting`, `diff-gaussian-rasterization`, `simple-knn` | "Gaussian-Splatting License" (Inria / MPII) | Uso consentito solo per ricerca non commerciale. Per l'uso commerciale serve un accordo con i titolari. |
| `hustvl/4DGaussians` (rif. [4] della guida) | Apache-2.0 | I sottomoduli sono `simple-knn` (Inria), un rasterizzatore derivato da quello Inria e i viewer SIBR, quindi valgono i vincoli Inria. |
| `JonathonLuiten/Dynamic3DGaussians` | MIT | Il file di licenza dice che il codice richiede rasterizzatore e parti di codice Inria, con la loro licenza. |
| `yangzf-1023/4C4D` (rif. [5] della guida) | MIT | L'installazione richiede componenti esterni (per esempio MAtCha) le cui licenze non sono state verificate. |

In pratica qualunque progetto che installi `diff-gaussian-rasterization` o
`simple-knn` è escluso, qualunque cosa dica il suo file LICENSE. I metodi
descritti negli articoli si possono riscrivere, come fa questo repository su
gsplat. Il loro codice no.

Altre cose da non aggiungere senza un controllo:

- Modelli corporei parametrici (SMPL, SMPL-X e derivati). Sono gratuiti solo
  per ricerca, e per l'uso commerciale serve una licenza a pagamento. Questo
  progetto non ne usa.
- Pesi di reti per segmentazione, matting o stima di profondità. La licenza
  dei pesi è spesso diversa da quella del codice. Qui le maschere si ottengono
  confrontando con il set vuoto, senza reti. Se si aggiunge un modello, in
  questa pagina va scritta la licenza del codice e quella dei pesi.
- Dataset pubblici. Molti dataset multicamera sono rilasciati per sola
  ricerca. Per le prove interne il repository genera dati sintetici
  (`mister4d demo`). Per ogni altro dataset va letta prima la licenza.

## 4. Che cosa resta da controllare

1. Brevetti. Una licenza software copre il codice e non i brevetti sui
   metodi. Prima di offrire a clienti un prodotto o un servizio basato su
   Gaussian Splatting conviene una verifica con un consulente.
2. Codec. I video usano i codec disponibili in FFmpeg (H.264 se presente). La
   distribuzione commerciale di contenuti o software con codec brevettati ha
   regole proprie.
3. Diritti sulle riprese. Le registrazioni di persone richiedono liberatorie
   e un trattamento conforme al GDPR.
4. Titolare. Nel file LICENSE il titolare è indicato come «MISTER Smart
   Innovation». Va controllato che sia la denominazione esatta della società.

## 5. Regole per chi sviluppa

- Non si copia codice da repository di ricerca. Si rilegge l'articolo e si
  riscrive.
- Ogni nuova dipendenza va aggiunta alla tabella del §2 con la licenza letta
  sul file LICENSE del pacchetto, controllando anche sottomoduli e dipendenze
  indirette. L'etichetta su GitHub non basta.
- Per controllare l'ambiente installato:

```bash
pip install pip-licenses
```

```bash
pip-licenses --from=mixed --order=license
```
