# Controlli fatti senza GPU

24/11/2026. Macchina: container Linux in cloud, 4 CPU, 15 GB di RAM, nessuna
GPU, nessun accesso alla workstation. Commit di `main` controllato: `a9f335f`.

Questa nota non contiene misure di qualità. Dice che cosa è stato possibile
controllare da qui prima di passare alla workstation.

## Verificato

1. **I pacchetti di `scripts/install.ps1` esistono per Windows e Python 3.10.**
   - `torch-2.4.1+cu124-cp310-cp310-win_amd64.whl` è nell'elenco di
     `https://download.pytorch.org/whl/cu124`.
   - `gsplat-1.5.3+pt24cu124-cp310-cp310-win_amd64.whl` è nell'elenco di
     `https://docs.gsplat.studio/whl/pt24cu124`. L'ho scaricato e aperto:
     contiene `gsplat/csrc.pyd` (71 MB), cioè il codice CUDA già compilato.
     Visual Studio e CUDA Toolkit quindi non servono.
   - Le dipendenze dichiarate da quel pacchetto sono `ninja`, `numpy`,
     `jaxtyping`, `rich`, `torch`. Lo script le installa tutte prima, quindi
     pip non deve cercarle nell'indice di gsplat, che contiene solo gsplat.
     Provato con pip 26.3 su Linux e Python 3.10, in un ambiente con quelle
     dipendenze e senza gsplat: `pip install --dry-run gsplat==1.5.3
     --index-url https://docs.gsplat.studio/whl/pt24cu124` risponde «Would
     install gsplat-1.5.3+pt24cu124».
   - Con `numpy<2` su Windows e Python 3.10 il resto si risolve così:
     numpy 1.26.4, scipy 1.15.3, opencv-python-headless 4.11.0.86,
     pyyaml 6.0.4, pytest 9.2.0, jaxtyping 0.3.7, rich 15.0.0, ninja 1.13.2.
     Calcolato con `uv pip compile --python-platform windows --python-version 3.10`.
     pip usa un altro risolutore: le versioni possono differire di poco.

2. **pytest passa con quelle versioni**, su Linux e CPU: Python 3.10.20,
   torch 2.4.1+cpu, numpy 1.26.4, scipy 1.15.3, opencv 4.11.0, gsplat 1.5.3.
   72 passati, 2 saltati, 8 minuti. I due saltati: il confronto tra gsplat e
   il rasterizzatore di riferimento (`tests/test_rasterizer.py`, serve una GPU
   CUDA) e il lettore C# per Unity (manca mono).

   Passa anche nell'ambiente già presente nel container: Python 3.12,
   torch 2.15, numpy 2.5, opencv 5.1. Stessi 72 passati e 2 saltati, 6 minuti.

3. **I comandi dei punti 2 e 3 girano**, su CPU e in formato minuscolo
   (160×90, 4 frame, 150 passi): `make_benchmark.py dataset`, `train` con le
   cinque camere `ver` escluse, `quality_check.py`, `contact_sheet.py`,
   `mister4d eval`, tutte le opzioni `--set` dell'elenco, `init --frame-step 3
   --exclude …` e la catena dai video (`demo --subject human --run`). Li ha
   lanciati `prove_gpu.py`, lo script di questa cartella, che così è
   collaudato a sua volta. I numeri di quel giro non sono qui: a quella
   dimensione non dicono nulla.

   Due cose utili da quel giro. `lambda_ray_spread` funziona con il
   rasterizzatore di riferimento. Lo stesso seme dà lo stesso risultato: due
   training uguali hanno dato misure identiche. Su GPU non sarà così, quindi
   tra le prove c'è `rumore` (stesse opzioni, altro seme) per sapere quanto
   oscillano le misure da sole.

## Non verificato

- `install.ps1` non è stato eseguito e nemmeno controllato con PowerShell: è
  stato solo letto. La prova vera è sulla workstation.
- Lo scaricamento di PyTorch per Windows: da qui si vede l'elenco, non il
  file.
- `mister4d doctor` su GPU, gsplat su CUDA, il training su due GPU.
- Tempi e memoria.
- `lambda_ray_spread` con gsplat.
- Ogni misura di qualità a 960×540: aghi, opacità mediana, spill, soft,
  thick, floaters, sfarfallio.
- FFmpeg di winget con la demo dai video.

## Notato leggendo il codice

Sono ipotesi da controllare sulla workstation, non difetti accertati.

1. **`lambda_ray_spread` e gsplat.** Il training passa 5 canali (colore,
   profondità, profondità al quadrato). Con `render_mode="RGB+ED"` gsplat ne
   aggiunge uno: 6. gsplat 1.5.3 disegna 1, 2, 3, 4, 5, 8, 9, 16… canali e
   porta gli altri numeri alla potenza di due successiva aggiungendo zeri
   (`rasterize_to_pixels` in `gsplat/cuda/_wrapper.py`). 6 diventano 8.
   Dovrebbe funzionare, con un po' più di memoria e di tempo.

2. **Se il codice CUDA non si carica, gsplat nasconde il motivo.** Scrive
   «gsplat: No CUDA toolkit found. gsplat will be disabled.» e il primo
   render fallisce con un errore su `NoneType`. `doctor` riporta
   quest'ultimo. L'errore vero si vede con:

   ```bat
   .venv\Scripts\python -c "import torch; from gsplat import csrc"
   ```

   Il pacchetto è etichettato `pt24` e lo script installa PyTorch 2.4.1. Se
   il caricamento fallisce, la prima prova è `torch==2.4.0`.

3. **gsplat usa `packaging` senza dichiararlo.** `gsplat/cuda/_backend.py`
   lo importa, ma non è tra le sue dipendenze. Oggi arriva insieme a pytest
   (`.[dev]`), quindi con `install.ps1` e `install.sh` c'è. Un'installazione
   senza `[dev]` fallirebbe al primo render su GPU. Proposta per `main`:
   aggiungere `packaging` alle dipendenze in `pyproject.toml` e alla tabella
   di `docs/LICENZE.md` (licenza Apache-2.0 oppure BSD-2-Clause).

4. **Cartelle con lettere accentate su Windows.** Le immagini si leggono e
   si scrivono con `cv2.imread` e `cv2.imwrite`. Su Windows OpenCV di solito
   non apre percorsi con caratteri fuori dalla code page del sistema:
   `imread` restituisce `None` senza errore e `imwrite` non scrive. Da
   provare con una cartella di progetto come `C:\Prove\Università`. Il
   rimedio noto è `cv2.imdecode(np.fromfile(…))` e `cv2.imencode(…).tofile(…)`.
   Per queste prove conviene un percorso senza accenti.

5. **`bench/` sporcava il checkout.** Non era in `.gitignore`, quindi dopo il
   punto 2a `git status` mostrava la cartella. Corretto su `main` (commit
   `3d02fc1`).

6. **I run si misurano dove sono nati.** `quality_check.py`,
   `contact_sheet.py` e `mister4d eval` leggono dal checkpoint il percorso
   assoluto del dataset. Se `bench\D` viene spostato serve `--data`.

7. **Le viste di `contact_sheet.py` con `--steps 6`.** Con le camere di
   training a −75°, −25°, 25°, 75° le sei viste cadono a circa −118°, −71°,
   −23°, 24°, 71°, 119°. Quattro sono a 2°–7° da una camera e due sono 40°
   oltre l'arco. Nessuna cade a metà tra due camere. Per quei punti di vista
   vale `sheet.png`, che usa le camere `ver01`…`ver03`. Con `--steps 7` una
   vista cade a 0°, a 25° dalle camere vicine.

## Ambiente di questi controlli

| | Versioni di Windows | Ambiente del container |
|---|---|---|
| Python | 3.10.20 | 3.12.3 |
| PyTorch | 2.4.1+cpu | 2.15.0+cu132, senza GPU |
| gsplat | 1.5.3 (da PyPI, senza codice compilato) | 1.5.3 |
| NumPy | 1.26.4 | 2.5.5 |
| SciPy | 1.15.3 | 1.18.1 |
| OpenCV | 4.11.0 | 5.1.0 |
| pytest | 72 passati, 2 saltati | 72 passati, 2 saltati |
