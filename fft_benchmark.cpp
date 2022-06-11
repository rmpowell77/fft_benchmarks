#include <cmath>
#include <complex>
#include <fftw3.h>
#include <nanobench.h>
#include <iostream>
#include <numbers>
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
  for (auto i : { 7, 8, 9, 10, 11, 12}) {
    auto fft = FFT(i);
    auto in = std::vector<FFT::complex>(fft.N);
    for (auto i = 0; i < fft.N; ++i) {
      in[i] = cos(i * 2 * std::numbers::pi/fft.N);
    }
    ankerl::nanobench::Bench().run(std::string("fftw ") + std::to_string(fft.N) , [&] {
        fft.execute();
    });
  }
}
