#include <Accelerate/Accelerate.h>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <iostream>
#include <nanobench.h>
#include <numbers>
#include <numeric>
#include <vector>

struct FFT {
  using complex = std::complex<double>;
  FFT(int log2n)
      : log2n(log2n), N(1 << log2n),
        in(static_cast<fftw_complex *>(fftw_malloc(sizeof(fftw_complex) * N))),
        out(static_cast<fftw_complex *>(fftw_malloc(sizeof(fftw_complex) * N))),
        p(fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE)) {}

  ~FFT() {
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
  }

  void fill_input(std::vector<complex> const &input) {
    std::copy(input.begin(), input.end(), reinterpret_cast<complex *>(in));
  }

  std::vector<complex> fetch_output() const {
    auto output = reinterpret_cast<complex *>(out);
    return {output, output + N};
  }

  void execute() { fftw_execute(p); }

  int log2n{};
  int N{};
  fftw_complex *in{};
  fftw_complex *out{};
  fftw_plan p{};
};

struct vDSP_FFT {
  using complex = DSPDoubleComplex;
  vDSP_FFT(int log2n)
      : log2n(log2n), N(1 << log2n), in(N), out(N),
        setup(vDSP_DFT_Interleaved_CreateSetupD(
            nullptr, N, vDSP_DFT_FORWARD,
            vDSP_DFT_Interleaved_ComplextoComplex)) {}

  ~vDSP_FFT() { vDSP_DFT_Interleaved_DestroySetupD(setup); }

  void fill_input(std::vector<complex> const &input) {
    std::copy(input.begin(), input.end(), in.begin());
  }

  std::vector<complex> fetch_output() const { return out; }

  std::vector<DSPDoubleComplex>
  convert(std::vector<DSPDoubleComplex> const &input) {
    fill_input(input);
    execute();
    return fetch_output();
  }

  void execute() {
    vDSP_DFT_Interleaved_ExecuteD(setup, in.data(), out.data());
  }

  int log2n{};
  int N{};
  std::vector<DSPDoubleComplex> in;
  std::vector<DSPDoubleComplex> out;
  vDSP_DFT_Interleaved_SetupD setup;
};

std::ostream &operator<<(std::ostream &os, vDSP_FFT::complex const &c) {
  return os << "(" << c.real << "," << c.imag << ")";
}
int main() {
  auto fft = FFT(4);

  auto in = std::vector<FFT::complex>(fft.N);
  for (auto i = 0; i < fft.N; ++i) {
    in[i] = cos(i * 2 * std::numbers::pi / fft.N);
  }

  std::cout << "Input: ";
  std::copy(in.begin(), in.end(),
            std::ostream_iterator<FFT::complex>(std::cout, ", "));
  std::cout << "\n";

  fft.fill_input(in);
  fft.execute();
  auto out = fft.fetch_output();

  std::cout << "\nOutput: ";
  std::copy(out.begin(), out.end(),
            std::ostream_iterator<FFT::complex>(std::cout, ", "));
  std::cout << "\n";

  // let's check out the error between the two
  auto vdsp_fft = vDSP_FFT(4);
  auto vdsp_in = std::vector<vDSP_FFT::complex>(fft.N);
  for (auto i = 0; i < fft.N; ++i) {
    vdsp_in[i].real = in[i].real();
  }
  vdsp_fft.fill_input(vdsp_in);
  vdsp_fft.execute();
  auto vdsp_out = vdsp_fft.fetch_output();
  auto result = std::inner_product(
      out.begin(), out.end(), vdsp_out.begin(), 0.0, std::plus<double>(),
      [](auto a, auto b) {
        auto r = std::complex<double>(a.real() - b.real, a.imag() - b.imag);
        return std::abs(r);
      });
  std::cout << "\nTotal error: " << result << "\n";

  for (auto i : {7, 8, 9, 10, 11, 12}) {
    auto fft = FFT(i);
    auto in = std::vector<FFT::complex>(fft.N);
    for (auto i = 0; i < fft.N; ++i) {
      in[i] = cos(i * 2 * std::numbers::pi / fft.N);
    }
    ankerl::nanobench::Bench().run(std::string("fftw ") + std::to_string(fft.N),
                                   [&] { fft.execute(); });
  }
  for (auto i : {7, 8, 9, 10, 11, 12}) {
    auto fft = vDSP_FFT(i);
    auto in = std::vector<vDSP_FFT::complex>(fft.N);
    for (auto i = 0; i < fft.N; ++i) {
      in[i].real = cos(i * 2 * std::numbers::pi / fft.N);
    }
    ankerl::nanobench::Bench().run(std::string("accelerate ") +
                                       std::to_string(fft.N),
                                   [&] { fft.execute(); });
  }
}
