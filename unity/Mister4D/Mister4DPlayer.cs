// MISTER 4D for Unity: plays a scene written by "mister4d export --format engine".
//
// Add the component to an empty GameObject and give it the name of the folder
// of the export inside StreamingAssets. The object is the subject: move, turn
// and scale it as any other. The scene keeps its measures (metres); its
// vertical axis, z in the capture, becomes y here.
//
// How it draws: a mesh of quads, one per visible Gaussian. Every frame the
// component chooses the Gaussians visible at that instant, sorts them back to
// front for the viewer and writes their order in a small texture; the shader
// does the rest (Resources/MisterSplat4D.shader). Only the chunk being shown
// and the next one are in memory, whatever the length of the take.

using System;
using System.Collections;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.Rendering;

namespace Mister4D
{
    /// <summary>scene.json of an engine export.</summary>
    [Serializable]
    public class SceneInfo
    {
        public string format;
        public int version;
        public int num_frames;
        public float fps;
        public float min_opacity;
        public int gaussians;
        public int alive_max;
        public float[] bbox_min;
        public float[] bbox_max;
        public ChunkInfo[] chunks;
    }

    [AddComponentMenu("MISTER 4D/Player")]
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
    public class Mister4DPlayer : MonoBehaviour
    {
        [Tooltip("Folder of the export inside StreamingAssets, or a full path, or a URL")]
        public string folder = "scena";
        public bool playOnStart = true;
        public bool loop = true;
        [Range(0.1f, 2f)] public float speed = 1f;
        [Range(0.2f, 1.5f)] public float splatScale = 1f;
        [Tooltip("The camera whose direction decides the drawing order (default: the main camera)")]
        public Camera viewer;

        /// <summary>True when the frame being asked for is on screen.</summary>
        public bool IsReady { get; private set; }
        public bool IsPlaying { get; private set; }
        public int Frame { get; private set; }
        public int FrameCount { get { return scene != null ? scene.num_frames : 0; } }
        /// <summary>Gaussians drawn in the last frame: what the device has to cope with.</summary>
        public int VisibleNow { get; private set; }
        /// <summary>What went wrong, if anything did.</summary>
        public string Error { get; private set; }

        const int TexWidth = 2048;  // Gaussians per row of the data textures

        static readonly int ALoId = Shader.PropertyToID("_ALo"), AHiId = Shader.PropertyToID("_AHi"),
            BId = Shader.PropertyToID("_B"), CId = Shader.PropertyToID("_C"), DId = Shader.PropertyToID("_D"),
            EId = Shader.PropertyToID("_E"), OrderId = Shader.PropertyToID("_Order"),
            TexWidthId = Shader.PropertyToID("_TexWidth"), CountId = Shader.PropertyToID("_Count"),
            BoxMinId = Shader.PropertyToID("_BoxMin"), BoxSizeId = Shader.PropertyToID("_BoxSize"),
            TimeRangeId = Shader.PropertyToID("_TimeRange"), VelMaxId = Shader.PropertyToID("_VelMax"),
            LifeId = Shader.PropertyToID("_Life"), LogScaleId = Shader.PropertyToID("_LogScale"),
            SplatTimeId = Shader.PropertyToID("_SplatTime"), MinOpacityId = Shader.PropertyToID("_MinOpacity"),
            SplatScaleId = Shader.PropertyToID("_SplatScale");

        // A chunk on the GPU. Its bytes stay here too: the choice of what to draw reads them.
        sealed class Loaded
        {
            public Chunk data;
            public Texture2D aLo, aHi, b, c, d, e;
        }

        // The textures of a chunk, laid out away from the main thread.
        sealed class Prepared
        {
            public Chunk chunk;
            public int height;
            public byte[] aLo, aHi, b, c, d, e;

            public static Prepared Build(ChunkInfo info, byte[] packed, float minOpacity)
            {
                byte[] raw = Chunk.Gunzip(packed);
                var chunk = new Chunk(info, raw, minOpacity);
                int n = info.count, height = Math.Max(1, (n + TexWidth - 1) / TexWidth), size = TexWidth * height * 4;
                var p = new Prepared
                {
                    chunk = chunk, height = height, aLo = new byte[size], aHi = new byte[size],
                    b = new byte[size], c = new byte[size], d = new byte[size], e = new byte[size],
                };
                for (int i = 0; i < 4 * n; i++)  // 16-bit values as two bytes: formats every device has
                {
                    p.aLo[i] = raw[2 * i];
                    p.aHi[i] = raw[2 * i + 1];
                }
                Buffer.BlockCopy(raw, 8 * n, p.b, 0, 4 * n);
                Buffer.BlockCopy(raw, 12 * n, p.c, 0, 4 * n);
                Buffer.BlockCopy(raw, 16 * n, p.d, 0, 4 * n);
                for (int i = 0, from = 20 * n; i < n; i++, from += 3)
                {
                    p.e[4 * i] = raw[from];
                    p.e[4 * i + 1] = raw[from + 1];
                    p.e[4 * i + 2] = raw[from + 2];
                    p.e[4 * i + 3] = 255;
                }
                return p;
            }
        }

        SceneInfo scene;
        Loaded[] loaded;
        bool[] loading;
        int bound = -1;
        float clock;
        Material material;
        Mesh mesh;
        MeshRenderer meshRenderer;
        Texture2D orderTexture;
        byte[] orderBytes;
        int[] order;
        SortBuffers buffers;

        void Start()
        {
            meshRenderer = GetComponent<MeshRenderer>();
            meshRenderer.enabled = false;
            StartCoroutine(LoadScene());
        }

        // ------------------------------------------------------------ control

        public void Play() { IsPlaying = true; }
        public void Pause() { IsPlaying = false; }

        public void Seek(int frame)
        {
            if (scene == null) return;
            clock = Mathf.Clamp(frame, 0, scene.num_frames - 1);
        }

        // ------------------------------------------------------------ loading

        // Where the export is, as a URL: on a headset StreamingAssets is inside the application package.
        string Root()
        {
            bool absolute = folder.Contains("://") || System.IO.Path.IsPathRooted(folder);
            string path = absolute ? folder : Application.streamingAssetsPath + "/" + folder;
            return path.Contains("://") ? path : new Uri(path).AbsoluteUri;
        }

        void Fail(string message)
        {
            Error = message;
            Debug.LogError("MISTER 4D: " + message, this);
        }

        IEnumerator LoadScene()
        {
            using (UnityWebRequest request = UnityWebRequest.Get(Root() + "/scene.json"))
            {
                yield return request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success)
                {
                    Fail("cannot read " + request.url + ": " + request.error);
                    yield break;
                }
                scene = JsonUtility.FromJson<SceneInfo>(request.downloadHandler.text);
            }
            if (scene == null || scene.format != "mister4d-engine" || scene.version != 1 || scene.chunks == null
                || scene.chunks.Length == 0)
            {
                scene = null;
                Fail("the folder '" + folder + "' is not an engine export of mister4d (version 1)");
                yield break;
            }
            Shader shader = Resources.Load<Shader>("MisterSplat4D");
            if (shader == null || !shader.isSupported)
            {
                scene = null;
                Fail("the shader MisterSplat4D is missing or not supported by this device");
                yield break;
            }

            int largest = 0;
            foreach (ChunkInfo c in scene.chunks) largest = Mathf.Max(largest, c.count);
            // A little more than the exporter counted: it rounds differently at the edge of visibility.
            int slots = Mathf.Min(largest, scene.alive_max + scene.alive_max / 10 + 1024);
            int orderHeight = Mathf.Max(1, (slots + TexWidth - 1) / TexWidth);
            order = new int[slots];
            orderBytes = new byte[TexWidth * orderHeight * 4];
            buffers = new SortBuffers(largest);
            orderTexture = DataTexture(orderHeight);

            mesh = BuildQuads(slots);
            Vector3 lo = new Vector3(scene.bbox_min[0], scene.bbox_min[2], scene.bbox_min[1]);
            Vector3 hi = new Vector3(scene.bbox_max[0], scene.bbox_max[2], scene.bbox_max[1]);
            mesh.bounds = new Bounds((lo + hi) * 0.5f, hi - lo + Vector3.one);  // room for the motion
            GetComponent<MeshFilter>().sharedMesh = mesh;

            material = new Material(shader) { name = "MISTER 4D", hideFlags = HideFlags.HideAndDontSave };
            material.SetTexture(OrderId, orderTexture);
            material.SetFloat(TexWidthId, TexWidth);
            material.SetFloat(MinOpacityId, scene.min_opacity);
            material.SetFloat(CountId, 0f);
            meshRenderer.sharedMaterial = material;
            meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
            meshRenderer.receiveShadows = false;
            meshRenderer.lightProbeUsage = LightProbeUsage.Off;
            meshRenderer.reflectionProbeUsage = ReflectionProbeUsage.Off;

            loaded = new Loaded[scene.chunks.Length];
            loading = new bool[scene.chunks.Length];
            IsPlaying = playOnStart;
        }

        IEnumerator LoadChunk(int index)
        {
            loading[index] = true;
            ChunkInfo info = scene.chunks[index];
            byte[] packed;
            using (UnityWebRequest request = UnityWebRequest.Get(Root() + "/" + info.file))
            {
                yield return request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success)
                {
                    Fail("cannot read " + request.url + ": " + request.error);
                    yield break;
                }
                packed = request.downloadHandler.data;
            }
            float minOpacity = scene.min_opacity;
            Task<Prepared> work = Task.Run(() => Prepared.Build(info, packed, minOpacity));
            while (!work.IsCompleted) yield return null;
            if (work.IsFaulted)
            {
                Fail("chunk " + info.file + ": " + work.Exception.GetBaseException().Message);
                yield break;
            }
            Prepared p = work.Result;
            loaded[index] = new Loaded
            {
                data = p.chunk,
                aLo = DataTexture(p.height, p.aLo), aHi = DataTexture(p.height, p.aHi), b = DataTexture(p.height, p.b),
                c = DataTexture(p.height, p.c), d = DataTexture(p.height, p.d), e = DataTexture(p.height, p.e),
            };
            loading[index] = false;
        }

        // Numbers, not pictures: no colour conversion, no filtering, no smaller copies.
        static Texture2D DataTexture(int height, byte[] bytes = null)
        {
            var texture = new Texture2D(TexWidth, height, TextureFormat.RGBA32, false, true)
            {
                filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp, anisoLevel = 0,
                hideFlags = HideFlags.HideAndDontSave,
            };
            if (bytes != null)
            {
                texture.LoadRawTextureData(bytes);
                texture.Apply(false, true);
            }
            return texture;
        }

        // Quads in drawing order: x and y of a vertex are its corner, z the position of its quad.
        static Mesh BuildQuads(int slots)
        {
            var vertices = new Vector3[slots * 4];
            var indices = new int[slots * 6];
            for (int i = 0; i < slots; i++)
            {
                int v = 4 * i, k = 6 * i;
                vertices[v] = new Vector3(-1f, -1f, i);
                vertices[v + 1] = new Vector3(1f, -1f, i);
                vertices[v + 2] = new Vector3(-1f, 1f, i);
                vertices[v + 3] = new Vector3(1f, 1f, i);
                indices[k] = v; indices[k + 1] = v + 1; indices[k + 2] = v + 2;
                indices[k + 3] = v + 2; indices[k + 4] = v + 1; indices[k + 5] = v + 3;
            }
            var mesh = new Mesh { name = "MISTER 4D quads", indexFormat = IndexFormat.UInt32,
                                  hideFlags = HideFlags.HideAndDontSave };
            mesh.vertices = vertices;
            mesh.SetIndices(indices, MeshTopology.Triangles, 0, false);
            return mesh;
        }

        // ------------------------------------------------------------ playing

        int ChunkOf(int frame)
        {
            for (int i = scene.chunks.Length - 1; i > 0; i--)
                if (frame >= scene.chunks[i].first) return i;
            return 0;
        }

        bool Has(int index) { return loaded[index] != null; }

        // Ask for the chunk being shown and, when it is under way, for the one after it; free the others.
        void KeepChunks(int index)
        {
            ChunkInfo info = scene.chunks[index];
            int next = -1;
            if (Has(index) && scene.chunks.Length > 1 && Frame >= (info.first + info.last) / 2)
                next = index + 1 < scene.chunks.Length ? index + 1 : (loop ? 0 : -1);
            for (int i = 0; i < loaded.Length; i++)
            {
                bool wanted = i == index || i == next;
                if (wanted && !Has(i) && !loading[i]) StartCoroutine(LoadChunk(i));
                else if (!wanted && Has(i) && loaded.Length > 2) Release(i);
            }
        }

        void Release(int index)
        {
            Loaded chunk = loaded[index];
            if (chunk == null) return;
            foreach (Texture2D texture in new[] { chunk.aLo, chunk.aHi, chunk.b, chunk.c, chunk.d, chunk.e })
                Destroy(texture);
            loaded[index] = null;
            if (bound == index) bound = -1;
        }

        void Bind(int index)
        {
            Loaded chunk = loaded[index];
            ChunkInfo info = chunk.data.Info;
            material.SetTexture(ALoId, chunk.aLo);
            material.SetTexture(AHiId, chunk.aHi);
            material.SetTexture(BId, chunk.b);
            material.SetTexture(CId, chunk.c);
            material.SetTexture(DId, chunk.d);
            material.SetTexture(EId, chunk.e);
            material.SetVector(BoxMinId, new Vector4(info.bbox_min[0], info.bbox_min[1], info.bbox_min[2], 0f));
            material.SetVector(BoxSizeId, new Vector4(info.bbox_max[0] - info.bbox_min[0],
                info.bbox_max[1] - info.bbox_min[1], info.bbox_max[2] - info.bbox_min[2], 0f));
            material.SetVector(TimeRangeId, new Vector4(info.time_start, info.time_span, 0f, 0f));
            material.SetFloat(VelMaxId, info.vel_max);
            material.SetVector(LifeId, new Vector4(info.life_log_min, info.life_log_max, 0f, 0f));
            material.SetVector(LogScaleId, new Vector4(info.scale_log_min, info.scale_log_max, 0f, 0f));
            bound = index;
        }

        // After the cameras have moved, so that the order is right for this very frame.
        void LateUpdate()
        {
            if (scene == null || loaded == null || !string.IsNullOrEmpty(Error)) return;

            if (IsPlaying && scene.num_frames > 1)
            {
                float next = clock + Time.deltaTime * speed * scene.fps;
                if (next >= scene.num_frames)
                {
                    if (loop) next %= scene.num_frames;
                    else { next = scene.num_frames - 1; IsPlaying = false; }
                }
                if (Has(ChunkOf((int)next))) clock = next;  // otherwise wait: that part is still loading
            }
            Frame = Mathf.Clamp((int)clock, 0, scene.num_frames - 1);  // the instants the scene was trained on
            int index = ChunkOf(Frame);
            KeepChunks(index);
            IsReady = Has(index);
            meshRenderer.enabled = IsReady;
            if (!IsReady) return;
            if (bound != index) Bind(index);

            Camera eye = viewer != null ? viewer : Camera.main;
            if (eye == null) return;
            // Unity is left-handed with y up, the capture right-handed with z up: swapping y and z
            // goes from one to the other.
            Vector3 forward = transform.InverseTransformDirection(eye.transform.forward);
            int n = loaded[index].data.SelectBackToFront(Frame, forward.x, forward.z, forward.y, order, buffers);
            for (int i = 0, k = 0; i < n; i++, k += 4)  // 24-bit index in r g b
            {
                int id = order[i];
                orderBytes[k] = (byte)id;
                orderBytes[k + 1] = (byte)(id >> 8);
                orderBytes[k + 2] = (byte)(id >> 16);
            }
            orderTexture.LoadRawTextureData(orderBytes);
            orderTexture.Apply(false, false);
            material.SetFloat(CountId, n);
            material.SetFloat(SplatTimeId, Frame);
            material.SetFloat(SplatScaleId, splatScale);
            VisibleNow = n;
        }

        void OnDestroy()
        {
            if (loaded != null)
                for (int i = 0; i < loaded.Length; i++) Release(i);
            if (orderTexture != null) Destroy(orderTexture);
            if (material != null) Destroy(material);
            if (mesh != null) Destroy(mesh);
        }
    }
}
