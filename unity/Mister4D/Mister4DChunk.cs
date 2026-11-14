// MISTER 4D for Unity: one chunk of a scene, as written by "mister4d export --format engine".
//
// This file does not use the Unity API on purpose: it is the part that reads
// the bytes, decides which Gaussians can be seen at an instant and sorts them
// back to front, and it is compiled and run outside Unity against the
// exporter (see README.md).
//
// Layout of a chunk (after gunzip), N Gaussians, little endian, 23 bytes each:
//   A  uint16 x4   position x y z in the box of the chunk, time
//   B  uint8  x4   velocity x y z (signed, 128 = still; the code is squared), life
//   C  uint8  x4   rotation quaternion w x y z
//   D  uint8  x4   log-scale x y z, peak opacity
//   E  uint8  x3   colour r g b
// The long-lived Gaussians come first; the others are sorted by time.

using System;
using System.IO;
using System.IO.Compression;

namespace Mister4D
{
    /// <summary>How to turn the integers of a chunk back into metres and frames.</summary>
    [Serializable]
    public class ChunkInfo
    {
        public string file;
        public int bytes;
        public int first;          // first frame this chunk is used for
        public int last;           // one past the last
        public int count;          // Gaussians in the chunk
        public int long_lived;     // how many at the start are always candidates
        public float reach;        // frames around an instant within which the others can be visible (-1: all)
        public float[] bbox_min;   // metres, capture frame: right-handed, z up
        public float[] bbox_max;
        public float time_start;   // frames
        public float time_span;
        public float vel_max;      // metres per frame at full scale
        public float life_log_min; // log of the shortest and of the longest life, in frames
        public float life_log_max;
        public float scale_log_min;
        public float scale_log_max;
        public int alive_max;      // the most that are visible in one frame
    }

    /// <summary>Working arrays of the sort, shared by all the chunks of a scene.</summary>
    public sealed class SortBuffers
    {
        public readonly int[] Ids;
        public readonly float[] Depth;
        public readonly ushort[] Keys;
        public readonly int[] Counts = new int[65536];

        public SortBuffers(int largestChunk)
        {
            Ids = new int[largestChunk];
            Depth = new float[largestChunk];
            Keys = new ushort[largestChunk];
        }
    }

    public sealed class Chunk
    {
        public const int BytesPerGaussian = 23;
        public const int Always = 255;  // life code of a Gaussian that does not depend on time

        public readonly ChunkInfo Info;
        public readonly byte[] Raw;      // the decompressed bytes
        public readonly float[] Reach;   // frames around its instant in which each Gaussian is visible

        public static byte[] Gunzip(byte[] packed)
        {
            using (var input = new GZipStream(new MemoryStream(packed), CompressionMode.Decompress))
            using (var output = new MemoryStream(packed.Length * 2))
            {
                input.CopyTo(output);
                return output.ToArray();
            }
        }

        public Chunk(ChunkInfo info, byte[] raw, float minOpacity)
        {
            if (raw.Length != BytesPerGaussian * info.count)
                throw new InvalidDataException("chunk " + info.file + ": " + raw.Length + " bytes, expected "
                                               + BytesPerGaussian * info.count);
            Info = info;
            Raw = raw;
            int n = info.count, b = 8 * n, d = 16 * n;
            Reach = new float[n];
            for (int i = 0; i < n; i++)
            {
                int code = raw[b + 4 * i + 3];
                float peak = raw[d + 4 * i + 3] / 255f;
                if (code == Always) Reach[i] = float.PositiveInfinity;
                else if (peak <= minOpacity) Reach[i] = -1f;  // never visible
                else
                {
                    double sigma = Math.Exp(info.life_log_min + (info.life_log_max - info.life_log_min) * code / 254.0);
                    Reach[i] = (float)(sigma * Math.Sqrt(2.0 * Math.Log(peak / minOpacity)));
                }
            }
        }

        int TimeCode(int i)
        {
            int k = 8 * i + 6;
            return Raw[k] | (Raw[k + 1] << 8);
        }

        // First position in [from, to) of the time-sorted part whose time code is >= code.
        int FirstAtLeast(int from, int to, double code)
        {
            int lo = from, hi = to;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (TimeCode(mid) < code) lo = mid + 1; else hi = mid;
            }
            return lo;
        }

        /// <summary>
        /// The Gaussians visible at instant <paramref name="t"/> (frames), back to front along the direction
        /// (dx, dy, dz), given in the capture frame. Their indices go to <paramref name="order"/>.
        /// Returns how many.
        /// </summary>
        public int SelectBackToFront(float t, float dx, float dy, float dz, int[] order, SortBuffers s)
        {
            ChunkInfo info = Info;
            byte[] raw = Raw;
            int n = info.count, bOffset = 8 * n;
            double t0 = info.time_start, ts = info.time_span / 65535.0, vmax = info.vel_max;
            double bx = info.bbox_min[0], by = info.bbox_min[1], bz = info.bbox_min[2];
            double sx = (info.bbox_max[0] - bx) / 65535.0, sy = (info.bbox_max[1] - by) / 65535.0,
                   sz = (info.bbox_max[2] - bz) / 65535.0;

            // The long-lived ones are always looked at; of the others, only those around t.
            int from = info.long_lived, to = n;
            if (info.reach >= 0f)
            {
                from = FirstAtLeast(info.long_lived, n, (t - info.reach - t0) / ts);
                to = FirstAtLeast(from, n, (t + info.reach - t0) / ts + 1.0);
            }

            int[] ids = s.Ids;
            float[] depth = s.Depth;
            int m = 0;
            float lo = float.PositiveInfinity, hi = float.NegativeInfinity;
            for (int part = 0; part < 2; part++)
            {
                int start = part == 0 ? 0 : from, end = part == 0 ? info.long_lived : to;
                for (int i = start; i < end; i++)
                {
                    float r = Reach[i];
                    double dt = 0.0;
                    if (!float.IsPositiveInfinity(r))
                    {
                        dt = t - (t0 + TimeCode(i) * ts);
                        if (dt > r || -dt > r) continue;  // not born yet, or already gone
                    }
                    int a = 8 * i, b = bOffset + 4 * i;
                    double cx = (raw[b] - 128) / 127.0, cy = (raw[b + 1] - 128) / 127.0, cz = (raw[b + 2] - 128) / 127.0;
                    double move = vmax * dt;  // the velocity is the square of its code, with its sign
                    double x = bx + (raw[a] | (raw[a + 1] << 8)) * sx + cx * Math.Abs(cx) * move;
                    double y = by + (raw[a + 2] | (raw[a + 3] << 8)) * sy + cy * Math.Abs(cy) * move;
                    double z = bz + (raw[a + 4] | (raw[a + 5] << 8)) * sz + cz * Math.Abs(cz) * move;
                    float along = (float)(dx * x + dy * y + dz * z);
                    depth[m] = along;
                    ids[m++] = i;
                    if (along < lo) lo = along;
                    if (along > hi) hi = along;
                }
            }

            // Counting sort on 16 bits of depth, far ones first.
            if (m > order.Length) m = order.Length;
            float scale = 65535f / Math.Max(hi - lo, 1e-9f);
            int[] counts = s.Counts;
            ushort[] keys = s.Keys;
            Array.Clear(counts, 0, counts.Length);
            for (int i = 0; i < m; i++)
            {
                int key = 65535 - (int)((depth[i] - lo) * scale);
                if (key < 0) key = 0;
                keys[i] = (ushort)key;
                counts[key]++;
            }
            int total = 0;
            for (int i = 0; i < 65536; i++) { int c = counts[i]; counts[i] = total; total += c; }
            for (int i = 0; i < m; i++) order[counts[keys[i]]++] = ids[i];
            return m;
        }
    }
}
