/*
 * qkv_kernel.cu
 * 3値量子化された重みを用いた QKV 線形射影の最適化 CUDA カーネル
 *
 * 重み: {-1, 0, +1}  (int8 格納)
 * 活性化: INT8 量子化済み
 *

 */

#include <cuda_runtime.h>
#include <cstdint>

/* ── タイル・ブロックパラメータ ── */
#define BM   64     // ブロックあたりの出力行数
#define BN   64     // ブロックあたりの出力列数
#define BK   32     // K方向のタイルサイズ
#define TM    4     // スレッドあたりの出力行数
#define TN    4     // スレッドあたりの出力列数
#define PAD   4     // 共有メモリパディング (バンクコンフリクト回避)

// ブロック次元: (BN/TN, BM/TM) = (16, 16) = 256 スレッド
static_assert(BM % TM == 0, "BM must be divisible by TM");
static_assert(BN % TN == 0, "BN must be divisible by TN");

__global__ void ternary_gemm_kernel(
    const int8_t* __restrict__ A,   // (M, K)  量子化活性化
    const int8_t* __restrict__ W,   // (N, K)  3値量子化重み
    float*        __restrict__ C,   // (M, N)  出力
    const int M, const int N, const int K)
{
    /* ── 共有メモリ (パディング付き) ── */
    __shared__ int8_t As[BM][BK + PAD];
    __shared__ int8_t Ws[BN][BK + PAD];

    const int tx  = threadIdx.x;                // 0..15  (N方向)
    const int ty  = threadIdx.y;                // 0..15  (M方向)
    const int tid = ty * blockDim.x + tx;       // 0..255
    const int bm  = blockIdx.y * BM;            // ブロック M オフセット
    const int bn  = blockIdx.x * BN;            // ブロック N オフセット

    /* ── レジスタアキュムレータ初期化 ── */
    int acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0;

    /* ── K 方向タイルループ ── */
    for (int kOff = 0; kOff < K; kOff += BK) {

        /* ── A タイルを共有メモリへ協調ロード ──
         *    BM * BK = 2048 要素 / 256 スレッド = 8 要素/スレッド */
        #pragma unroll
        for (int i = tid; i < BM * BK; i += 256) {
            const int r  = i / BK;
            const int c  = i % BK;
            const int gr = bm + r;
            const int gc = kOff + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : int8_t(0);
        }

        /* ── W タイルを共有メモリへ協調ロード ── */
        #pragma unroll
        for (int i = tid; i < BN * BK; i += 256) {
            const int r  = i / BK;
            const int c  = i % BK;
            const int gr = bn + r;
            const int gc = kOff + c;
            Ws[r][c] = (gr < N && gc < K) ? W[gr * K + gc] : int8_t(0);
        }

        __syncthreads();

        /* ── タイル内演算 ── */
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 共有メモリ → レジスタ
            int8_t ra[TM], rw[TN];

            #pragma unroll
            for (int m = 0; m < TM; ++m)
                ra[m] = As[ty * TM + m][k];

            #pragma unroll
            for (int n = 0; n < TN; ++n)
                rw[n] = Ws[tx * TN + n][k];

            // 累積: w∈{-1,0,+1} なので int 乗算で分岐不要
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += static_cast<int>(ra[m]) * static_cast<int>(rw[n]);
        }

        __syncthreads();
    }

    /* ── 結果をグローバルメモリへ書き出し ── */
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int gr = bm + ty * TM + m;
        if (gr < M) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                const int gc = bn + tx * TN + n;
                if (gc < N)
                    C[gr * N + gc] = static_cast<float>(acc[m][n]);
            }
        }
    }
}

/*
 * ternary_qkv_gemm_cuda  —  ホスト側ラッパー
 */
void ternary_qkv_gemm_cuda(
    const int8_t* A, const int8_t* W, float* C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 block(BN / TN, BM / TM);   // (16, 16) = 256 スレッド
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    ternary_gemm_kernel<<<grid, block, 0, stream>>>(A, W, C, M, N, K);
}
