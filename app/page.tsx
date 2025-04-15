"use client";

import React, { useState } from "react";


export default function Home() {
  const [copiedIndex, setCopiedIndex] = useState(null);

  const content = [
    {
      heading: "ELEMENTARY SIGNAL",
      text: `import numpy as np
import matplotlib.pyplot as plt

#defining functions
def impulse(time , pos):
  dis_signal = np.zeros(time)
  dis_signal[pos] = 1
  return dis_signal

def unit(time):
  unit_signal = np.ones(time)
  return unit_signal

def ramp(time):
  ramp_signal = np.zeros(time)
  for i in range(time):
    ramp_signal[i] = i
  return ramp_signal

def sine(time , amp ,freq):
  t = np.arange(0 ,time+1)
  dis_signal = amp * np.sin(2*np.pi*freq*t)
  return dis_signal

def cos(time , amp ,freq):
  t = np.arange(0 ,time+1)
  dis_signal = amp * np.cos(2*np.pi*freq*t)
  return dis_signal

def exp(time , amp ,freq):
  t = np.arange(0 ,time+1)
  dis_signal = amp * np.exp(freq*t)
  return dis_signal

# calling functions
impulse_signal = impulse(25 , 6)
unit_signal = unit(11)
ramp_signal = ramp(15)
sine_signal = sine(80 , 1 , 0.9)
cos_signal = cos(80 , 1 , 0.9)
exp_signal = exp(80 , 1 , 0.9)

#plotting fuctions
plt.figure(figsize=(10,14))
plt.subplot(9,1,1)
plt.stem(impulse_signal)
plt.title('Impulse')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(9,1,2)
plt.stem(unit_signal)
plt.title('Unit')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(9,1,3)
plt.stem(ramp_signal)
plt.title('Ramp')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(9,1,4)
plt.stem(sine_signal)
plt.title('Sine signal discrete')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(9,1,5)
plt.plot(sine_signal)
plt.title('Sine signal continuous')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(9,1,6)
plt.stem(cos_signal)
plt.title('Cosine signal discrete')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(9,1,7)
plt.plot(cos_signal)
plt.title('Cosine signal continuous')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(9,1,8)
plt.stem(exp_signal)
plt.title('Exponential signal discrete')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(9,1,9)
plt.plot(exp_signal)
plt.title('Exponential signal continuous')
plt.xlabel('Time')
plt.ylabel('Amplitude')


# Add spacing between plots
plt.tight_layout(pad=2.5)
plt.show()`,
    },
    {
      heading: "LINEAR CONVOLUTION",
      text: `
      input_string_for_x_n = input("Enter the elements for x_n separated by spaces: ")
input_string_for_y_n = input("Enter the elements for y_n separated by spaces: ")

string_list_for_x_n = input_string_for_x_n.split()
string_list_for_y_n = input_string_for_y_n.split()

x_n = [int(num) for num in string_list_for_x_n]
y_n = [int(num) for num in string_list_for_y_n]

x_origin = int(input("Enter the origin index for x_n: "))
y_origin = int(input("Enter the origin index for y_n: "))

# x_origin = x_n[1]
# y_origin = y_n[0]

l = len(x_n)
m = len(y_n)

print(x_n)
print(y_n)
print(x_origin)
print(y_origin)

# Convolution numpy se
numpy_conv = np.convolve(x_n, y_n)
len_numpy_conv = len(numpy_conv)
print("Convolution using NumPy:", numpy_conv)
print("Length of NumPy convolution:", len_numpy_conv)
print("Origin of convolution present at index:" , x_origin+y_origin , "and the value present at that index is " ,numpy_conv[x_origin+y_origin] )

# Manual convolution(i.e using for loop)
manual_conv = []
for n in range(l + m - 1):
  sum_val = 0
  for k in range(l):
    if 0 <= n - k < m:
      sum_val += x_n[k] * y_n[n - k]
  manual_conv.append(sum_val)

print("Manual convolution:", manual_conv)
print("Length of manual convolution:", len(manual_conv))
print("Origin of manual convolution present at index :" , x_origin + y_origin , "and the value at that index is :" , manual_conv[x_origin+y_origin])
`,
    },
    {
      heading: "CIRCULAR CONVOLUTION",
      text: `
      x_n = [int(num) for num in input("Enter the elements for x_n separated by spaces: ").split()]
y_n = [int(num) for num in input("Enter the elements for y_n separated by spaces: ").split()]

l = len(x_n)
m = len(y_n)

print(x_n)
print(y_n)

N = max(l,m)
print(N)

# padding with zeros
if l < N:
    for i in range(N-l):
        x_n.append(0)

if m < N:
    for i in range(N-m):
        y_n.append(0)

manual_conv = []
for n in range(N):
    sum_val = 0
    for k in range(N):
        sum_val += x_n[k] * y_n[(n - k) % N]
    manual_conv.append(sum_val)

print(x_n)
print(y_n)
print("circular convolution:" , manual_conv)
`,
    },
    {
      heading: "DFT",
      text: `
      import numpy as np
import matplotlib.pyplot as plt

x_n = [int(num) for num in input("Enter the x_n separated by spaces: ").split()]
N = len(x_n)

# Initialize the twiddle factor matrix
twiddle_matrix = np.zeros((N, N), dtype=complex)

# Compute only unique twiddle factors using periodicity and symmetry
WN = np.exp(-2j * np.pi / N)  # Fundamental twiddle factor
twiddle_factors = {}

for k in range(N):
    for n in range(N):
        exponent = (k * n) % N  # Using periodicity W_N^(n+k) = W_N^k
        if exponent not in twiddle_factors:
            twiddle_factors[exponent] = WN**exponent
        twiddle_matrix[k, n] = twiddle_factors[exponent]

print("\nTwiddle Factor Matrix:")
print(twiddle_matrix)

# Compute DFT using the optimized twiddle matrix
X_k = np.dot(twiddle_matrix, x_n)

print("\nDFT of the sequence (Manual Calculation):")
for i, val in enumerate(X_k):
    print(f"X[{i}] = {val}")

# Compute DFT using NumPy's FFT for verification
X_k_fft = np.fft.fft(x_n)

print("\nDFT of the sequence (Using NumPy FFT):")
for i, val in enumerate(X_k_fft):
    print(f"X_fft[{i}] = {val}")

# Check if manual DFT matches NumPy's FFT
if np.allclose(X_k, X_k_fft):
    print("\n The manually computed DFT matches NumPy's FFT!")
else:
    print("\n The manually computed DFT does NOT match NumPy's FFT!")

# Plot Magnitude and Phase Spectrum
plt.figure(figsize=(12, 5))

# Magnitude Spectrum
plt.subplot(1, 2, 1)
plt.stem(range(N), np.abs(X_k), basefmt=" ", label="Manual DFT")
plt.stem(range(N), np.abs(X_k_fft), basefmt=" ", markerfmt='ro', linefmt='r', label="NumPy FFT")
plt.xlabel('Frequency Index (k)')
plt.ylabel('Magnitude')
plt.title('DFT Magnitude Spectrum')
plt.legend()
plt.grid()

# Phase Spectrum
plt.subplot(1, 2, 2)
plt.stem(range(N), np.angle(X_k), basefmt=" ", label="Manual DFT")
plt.stem(range(N), np.angle(X_k_fft), basefmt=" ", markerfmt='ro', linefmt='r', label="NumPy FFT")
plt.xlabel('Frequency Index (k)')
plt.ylabel('Phase (radians)')
plt.title('DFT Phase Spectrum')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
`,
    },
    {
      heading: "IDFT",
      text: `
      import numpy as np
import matplotlib.pyplot as plt

# Input DFT values
X_k = [complex(num) for num in input("Enter the X_k separated by spaces: ").split()]
N = len(X_k)

# Initialize the inverse twiddle factor matrix
inv_w_mat = np.zeros((N, N), dtype=complex)

# to compute only unique twiddle factor using properties
WN = np.exp(-2j * np.pi / N)  # twiddle factor
inv_w_fac = {}

for k in range(N):
    for n in range(N):
        peri_exp = (k * n) % N  # Using periodicity W_N^(n+k) = W_N^k
        symm_exp = (n*k) % N/2  # Using symmetric property here
        if peri_exp not in inv_w_fac:
            inv_w_fac[peri_exp] = WN**(-peri_exp)  # Conjugate twiddle for IDFT
            inv_w_mat[k, n] = inv_w_fac[peri_exp]
        elif symm_exp not in inv_w_fac:
            inv_w_fac[symm_exp] = -WN**(-symm_exp)  # Conjugate twiddle for IDFT
            inv_w_mat[k, n] = inv_w_fac[symm_exp]
        inv_w_mat[k, n] = inv_w_fac[peri_exp]

print("\nInverse Twiddle Factor Matrix:")
print(inv_w_mat)

# Compute IDFT using the optimized inverse twiddle matrix
x_n_manual = (1 / N) * np.dot(inv_w_mat, X_k)

print("\nIDFT of the sequence (Manual Calculation):")
for i, val in enumerate(x_n_manual):
    print(f"x[{i}] = {val}")

# to check whether manual one is correct or not
x_n_ifft = np.fft.ifft(X_k)

print("\nIDFT of the sequence (Using NumPy IFFT):")
for i, val in enumerate(x_n_ifft):
    print(f"x_ifft[{i}] = {val}")


# Plot Magnitude and Phase Spectrum of the IDFT result
plt.figure(figsize=(12, 5))

# Magnitude Spectrum of the IDFT result
plt.subplot(1, 2, 1)
plt.stem(range(N), np.abs(x_n_manual), basefmt=" ", label="Manual IDFT")
plt.stem(range(N), np.abs(x_n_ifft), basefmt=" ", markerfmt='ro', linefmt='r', label="NumPy IFFT")
plt.xlabel('Time Index (n)')
plt.ylabel('Magnitude')
plt.title('IDFT Magnitude Spectrum')
plt.legend()
plt.grid()

# Phase Spectrum of the IDFT result
plt.subplot(1, 2, 2)
plt.stem(range(N), np.angle(x_n_manual), basefmt=" ", label="Manual IDFT")
plt.stem(range(N), np.angle(x_n_ifft), basefmt=" ", markerfmt='ro', linefmt='r', label="NumPy IFFT")
plt.xlabel('Time Index (n)')
plt.ylabel('Phase (radians)')
plt.title('IDFT Phase Spectrum')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
`,
    },
    {
      heading: "OVERLAP ADD",
      text: `
      def circular_convolution(x, h, N):
    y = [0] * N
    for n in range(N):
        for m in range(N):
            y[n] += x[m] * h[(n - m) % N]
    return y


def linear_convolution(x, h):
    M, N = len(h), len(x)
    y = [0] * (M + N - 1)
    for i in range(N):
        for j in range(M):
            y[i + j] += x[i] * h[j]
    return y


def overlap_add(x, h, L):
    M = len(h)
    N = L + M - 1  # Length of each circular convolution
    h_padded = h + [0] * (L - 1)  # Pad h to length N

    num_blocks = (len(x) + L - 1) // L  # Number of blocks 12//3(len)
    output = [0] * (len(x) + M - 1)  # Output sequence 12(len)

    for i in range(num_blocks):
        start_idx = i * L
        end_idx = min(start_idx + L, len(x))
        x_block = x[start_idx:end_idx] + [0] * (N - (end_idx - start_idx))

        # Circular convolution
        y_block = circular_convolution(x_block, h_padded, N)

        # x_block
        print(f"Iteration {i+1}: x_block = {x_block}")
        # y_block
        print(f"Iteration {i+1}: y_block = {y_block}")

        # Overlap-Add step
        for j in range(N):
            if start_idx + j < len(output):
                output[start_idx + j] += y_block[j]

    return output


# User input
h = [int(num) for num in input("Enter the elements for h[x] separated by spaces: ").split()] #lenght -> input 3
x = [int(num) for num in input("Enter the elements for x[n] separated by spaces: ").split()] #length -> input 10
L = int(input("Enter L (block length for breaking x[n]): ")) #length -> input 3

# Overlap-Add method
res_overlap_add = overlap_add(x, h, L)
print("Filtered output sequence (Overlap-Add):", res_overlap_add)

# Verify using Linear Convolution
result_linear = linear_convolution(x, h)
print("Filtered output sequence (Linear Convolution):", result_linear)
`,
    },
    {
      heading: "OVERLAOP SAVE",
      text: `
      def circular_convolution(x, h, N):
    y = [0] * N
    for n in range(N):
        for m in range(N):
            y[n] += x[m] * h[(n - m) % N]
    return y


def circular_convolution(x, h, N):
    y = [0] * N
    for n in range(N):
        for m in range(N):
            y[n] += x[m] * h[(n - m) % N]
    return y

def linear_convolution(x, h):
    M, N = len(h), len(x)
    y = [0] * (M + N - 1)
    for i in range(N):
        for j in range(M):
            y[i + j] += x[i] * h[j]
    return y

def overlap_save(x, h, L):
    M = len(h)
    N = L + M - 1
    h_padded = h + [0] * (L - 1)  # Pad h to match N

    # Split x into overlapping blocks of size N
    x_padded = [0] * (M - 1) + x  # Prepend M-1 zeros
    num_blocks = (len(x_padded) + L - 1) // L
    output = []

    for i in range(num_blocks):
        start_idx = i * L
        end_idx = min(start_idx + N, len(x_padded))
        x_block = x_padded[start_idx:end_idx] + [0] * (N - (end_idx - start_idx))

        # circular convolution
        y_block = circular_convolution(x_block, h_padded, N)

        #x_block
        print(f"Iteration {i+1}: x_block = {x_block}")
        # y_block
        print(f"Iteration {i+1}: y_block = {y_block}")

        output.extend(y_block[M - 1:])  # Remove first M-1 values (overlap part)

    return output

# User input
h = [int(num) for num in input("Enter the elements for h[x] separated by spaces: ").split()]
x = [int(num) for num in input("Enter the elements for x[n] separated by spaces: ").split()]
L = int(input("Enter L (block length for breaking x[n]): "))

# Overlap-Save method
res_overlap_save = overlap_save(x, h, L)
print("Filtered output sequence (Overlap-Save):", res_overlap_save)

# Verify using Linear Convolution
result_linear = linear_convolution(x, h)
print("Filtered output sequence (Linear Convolution):", result_linear)
`,
    },
    {
      heading: "DIF FFT",
      text: `
      import numpy as np

def bit_reversal_indices(N):
    num_bits = int(np.log2(N))
    indices = np.arange(N)
    reversed_indices = np.zeros(N, dtype=int)

    for i in range(N):
        rev = 0
        num = i
        for _ in range(num_bits):
            rev = (rev << 1) | (num & 1)
            num >>= 1
        reversed_indices[i] = rev

    return reversed_indices

def radix2_dif_fft(x):
    """Compute the radix-2 DIF-FFT of sequence x using the Butterfly Algorithm (two loops)."""
    N = len(x)
    x = np.array(x, dtype=complex)

    # Butterfly computation (decimation in frequency)
    m = N
    while m >= 2:
        W_m = np.exp(-2j * np.pi / m)  # Twiddle factor
        for k in range(0, N, m):
            W = 1
            for j in range(m // 2):
                u = x[k + j]
                t = x[k + j + m // 2]
                x[k + j] = u + t
                x[k + j + m // 2] = (u - t) * W
                W *= W_m
        m //= 2

    # Bit-reversal reordering (after butterfly computation)
    reversed_indices = bit_reversal_indices(N)
    x = x[reversed_indices]

    return x

# Main execution
x_n = [int(num) for num in input("Enter the sequence of x_n separated by spaces: ").split()]
X_k = radix2_dif_fft(x_n)

print("DIF FFT Output:")
print(X_k)

# Compare with NumPy's FFT
X_k_builtin = np.fft.fft(x_n)

print("\nNumPy Built-in FFT Output:")
print(X_k_builtin)

# Check difference
print("\nDifference (Custom - Built-in):")
print(np.abs(X_k - X_k_builtin))

print("\nVerification:", np.allclose(X_k, X_k_builtin))
`,
    },
    {
      heading: "DIT FFT ",
      text: `
      import numpy as np

def reversed(N):
    num_bits = int(np.log2(N))
    rev_bits = np.zeros(N, dtype = int)
    
    for i in range(N):
        num = i
        rev = 0
        for _ in range(num_bits):
            rev = rev << 1 | num & 1
            num = num >> 1
        rev_bits[i] = rev
    return rev_bits


def radix(x):
    N = len(x)
    rev = reversed(N)
    x = x[rev]
    m = 2
    while m <= N:
        WN = np.exp(-2j * np.pi / m)
        for j in range(0,N,m):
            W = 1
            for k in range(m//2):
                u = x[k + j]
                t = x[k + j + m//2]*W
                x[k+j] = u + t
                x[k + j + m//2] = (u - t)
                W *= WN
        m = m*2
    return x




x = np.array([complex(num) for num in input().split()])
ans = radix(x)

built = np.fft.fft(x)
print(np.allclose(ans,built))
print(ans)
`,
    },
  ];

  const handleCopy = (text: any, index: any) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000); // Reset the button text after 2 seconds
    });
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-4">
      {content.map((item, index) => (
        <div
          key={index}
          className="group relative transition-all duration-200 ease-in-out transform hover:scale-[1.005]"
        >
          <div className="bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow duration-200 border border-gray-100">
            <div className="flex items-center justify-between p-5">
              <div className="pr-4">
                <h2 className="text-lg font-semibold text-gray-800 mb-1">
                  {item.heading}
                </h2>
              </div>
              
              <button
                onClick={() => handleCopy(item.text, index)}
                className={`flex items-center space-x-2 px-5 py-2.5 rounded-lg transition-colors duration-150 ${
                  copiedIndex === index 
                    ? "bg-green-100 text-green-700 hover:bg-green-200"
                    : "bg-blue-50 text-blue-600 hover:bg-blue-100"
                }`}
              >
                {copiedIndex === index ? (
                  <>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    <span>Copied!</span>
                  </>
                ) : (
                  <>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
                      <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
                    </svg>
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
          </div>
          
          {/* Subtle animated border effect when clicked */}
          {copiedIndex === index && (
            <div className="absolute inset-0 border-2 border-green-200 rounded-xl animate-ping-once" />
          )}
        </div>
      ))}
    </div>
  );
}
