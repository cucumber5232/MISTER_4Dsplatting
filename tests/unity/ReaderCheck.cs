// Runs the part of the Unity reader that does not need Unity (unity/Mister4D/Mister4DChunk.cs) on an
// exported chunk: test_unity.py compiles it with mono, when it is installed, and compares the outcome.
using System;
using System.Globalization;
using System.IO;
using Mister4D;

static class Program
{
    static int Main(string[] args)
    {
        var inv = CultureInfo.InvariantCulture;
        var info = new ChunkInfo { bbox_min = new float[3], bbox_max = new float[3] };
        float minOpacity = 0.02f;
        foreach (var line in File.ReadAllLines(args[0]))
        {
            var kv = line.Split('=');
            string k = kv[0], v = kv[1];
            switch (k)
            {
                case "file": info.file = v; break;
                case "count": info.count = int.Parse(v, inv); break;
                case "long_lived": info.long_lived = int.Parse(v, inv); break;
                case "first": info.first = int.Parse(v, inv); break;
                case "last": info.last = int.Parse(v, inv); break;
                case "reach": info.reach = float.Parse(v, inv); break;
                case "time_start": info.time_start = float.Parse(v, inv); break;
                case "time_span": info.time_span = float.Parse(v, inv); break;
                case "vel_max": info.vel_max = float.Parse(v, inv); break;
                case "life_log_min": info.life_log_min = float.Parse(v, inv); break;
                case "life_log_max": info.life_log_max = float.Parse(v, inv); break;
                case "min_opacity": minOpacity = float.Parse(v, inv); break;
                case "bbox_min": { var p = v.Split(','); for (int i = 0; i < 3; i++) info.bbox_min[i] = float.Parse(p[i], inv); break; }
                case "bbox_max": { var p = v.Split(','); for (int i = 0; i < 3; i++) info.bbox_max[i] = float.Parse(p[i], inv); break; }
            }
        }
        var chunk = new Chunk(info, Chunk.Gunzip(File.ReadAllBytes(info.file)), minOpacity);
        var buffers = new SortBuffers(info.count);
        var order = new int[info.count];
        using (var output = new BinaryWriter(File.Create(args[1])))
        {
            for (int i = 2; i + 3 < args.Length; i += 4)
            {
                float t = float.Parse(args[i], inv);
                int n = chunk.SelectBackToFront(t, float.Parse(args[i + 1], inv), float.Parse(args[i + 2], inv),
                                                float.Parse(args[i + 3], inv), order, buffers);
                output.Write(n);
                for (int k = 0; k < n; k++) output.Write(order[k]);
            }
        }
        return 0;
    }
}
