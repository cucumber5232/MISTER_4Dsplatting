// MISTER 4D for Unity: draws the Gaussians of a chunk at one instant.
//
// The same computation as the web player (src/mister4d/viewer/viewer.js): every
// Gaussian is stored once, with its instant, life and velocity, and is moved and
// faded here for the instant being shown. The mesh is a list of quads; the z of
// a vertex is the position of its quad in the drawing order, x and y its corner.
Shader "MISTER/Splat4D"
{
    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" "IgnoreProjector" = "True" }
        Pass
        {
            ZWrite Off
            ZTest LEqual
            Cull Off
            Blend One OneMinusSrcAlpha  // premultiplied colours, drawn back to front

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.5
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"

            // Blocks of the chunk, one Gaussian per texel (see Mister4DChunk.cs).
            Texture2D _ALo;    // low bytes of: position x y z, time
            Texture2D _AHi;    // high bytes of the same
            Texture2D _B;      // velocity x y z, life
            Texture2D _C;      // rotation w x y z
            Texture2D _D;      // log-scale x y z, peak opacity
            Texture2D _E;      // colour
            Texture2D _Order;  // which Gaussian each quad draws: 24-bit index in r g b
            float _TexWidth;
            float _Count;        // quads in use
            float3 _BoxMin;
            float3 _BoxSize;
            float2 _TimeRange;   // first instant and span of the chunk, in frames
            float _VelMax;       // metres per frame at full scale
            float2 _Life;        // log of the shortest and of the longest life, in frames
            float2 _LogScale;
            float _SplatTime;    // the instant being shown, in frames
            float _MinOpacity;
            float _SplatScale;

            struct appdata
            {
                float4 vertex : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR0;
                float2 offset : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            int3 Texel(float index)
            {
                float row = floor(index / _TexWidth);
                return int3((int)(index - row * _TexWidth + 0.5), (int)(row + 0.5), 0);
            }

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.pos = float4(0.0, 0.0, 2.0, 1.0);  // outside the view: nothing is drawn
                float slot = floor(v.vertex.z + 0.5);
                if (slot >= _Count) return o;

                float4 which = round(_Order.Load(Texel(slot)) * 255.0);
                int3 uv = Texel(which.r + which.g * 256.0 + which.b * 65536.0);
                float4 a = (round(_AHi.Load(uv) * 255.0) * 256.0 + round(_ALo.Load(uv) * 255.0)) / 65535.0;
                float4 b = round(_B.Load(uv) * 255.0);
                float4 d = _D.Load(uv);

                float dt = 0.0, fade = 1.0;
                if (b.a < 254.5)  // 255 = does not depend on time
                {
                    dt = _SplatTime - (_TimeRange.x + a.w * _TimeRange.y);
                    float life = exp(lerp(_Life.x, _Life.y, b.a / 254.0));
                    fade = exp(-0.5 * dt * dt / (life * life));
                }
                float opacity = d.a * fade;
                if (opacity < _MinOpacity) return o;

                // Position in the capture frame (metres, right-handed, z up). The velocity is the
                // square of its code, with its sign.
                float3 code = (b.rgb - 128.0) / 127.0;
                float3 p = _BoxMin + a.xyz * _BoxSize + code * abs(code) * _VelMax * dt;

                // Covariance in the capture frame: R S S^T R^T.
                float4 q = normalize(_C.Load(uv) * 2.0 - 1.0);  // w x y z
                float w = q.x, x = q.y, y = q.z, z = q.w;
                float3x3 R = float3x3(
                    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
                float3 s = _SplatScale * exp(lerp(_LogScale.xxx, _LogScale.yyy, d.rgb));
                float3x3 M = float3x3(R[0] * s, R[1] * s, R[2] * s);
                float3x3 cov = mul(M, transpose(M));

                // Capture frame -> object space of Unity (left-handed, y up): swap y and z.
                float3 pObject = p.xzy;
                float3x3 covObject = float3x3(
                    cov[0][0], cov[0][2], cov[0][1],
                    cov[2][0], cov[2][2], cov[2][1],
                    cov[1][0], cov[1][2], cov[1][1]);

                // To the space of the eye (it looks along -z).
                float3 eye = UnityObjectToViewPos(pObject);
                float depth = -eye.z;
                if (depth < 0.05) return o;
                float3x3 toEye = mul((float3x3)UNITY_MATRIX_V, (float3x3)unity_ObjectToWorld);
                float3x3 covEye = mul(toEye, mul(covObject, transpose(toEye)));

                // Project the covariance to the image, in pixels (EWA splatting).
                float2 focal = 0.5 * _ScreenParams.xy * float2(UNITY_MATRIX_P._m00, abs(UNITY_MATRIX_P._m11));
                float2 j0 = float2(focal.x / depth, focal.x * eye.x / (depth * depth));  // du/dx, du/dz
                float2 j1 = float2(focal.y / depth, focal.y * eye.y / (depth * depth));  // dv/dy, dv/dz
                float3 ju = float3(j0.x, 0.0, j0.y);
                float3 jv = float3(0.0, j1.x, j1.y);
                float3 cu = mul(covEye, ju);
                float3 cv = mul(covEye, jv);
                float ca = dot(ju, cu) + 0.3, cb = dot(ju, cv), cc = dot(jv, cv) + 0.3;

                // Principal axes of the 2D Gaussian.
                float mid = 0.5 * (ca + cc);
                float rad = sqrt(max(mid * mid - (ca * cc - cb * cb), 1e-8));
                float l1 = mid + rad, l2 = max(mid - rad, 0.05);
                float2 v1 = abs(cb) > 1e-6 ? normalize(float2(cb, l1 - ca)) : (ca >= cc ? float2(1.0, 0.0) : float2(0.0, 1.0));
                float2 v2 = float2(-v1.y, v1.x);
                float2 px = v.vertex.x * 3.33 * sqrt(l1) * v1 + v.vertex.y * 3.33 * sqrt(l2) * v2;  // pixels, y up

                float4 centre = UnityViewToClipPos(eye);
                float flip = UNITY_MATRIX_P._m11 < 0.0 ? -1.0 : 1.0;  // some targets are drawn upside down
                centre.xy += px * float2(2.0, 2.0 * flip) / _ScreenParams.xy * centre.w;
                o.pos = centre;
                o.offset = v.vertex.xy * 3.33;

                float3 rgb = _E.Load(uv).rgb;
            #ifndef UNITY_COLORSPACE_GAMMA
                rgb = GammaToLinearSpace(rgb);
            #endif
                o.color = float4(rgb, opacity);
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float alpha = min(i.color.a * exp(-0.5 * dot(i.offset, i.offset)), 0.999);
                clip(alpha - 1.0 / 255.0);
                return float4(i.color.rgb * alpha, alpha);
            }
            ENDCG
        }
    }
}
