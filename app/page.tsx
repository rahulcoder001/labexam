"use client"
import Link from "next/link";
import React, { useState } from "react";

export default function Home() {
  const [copiedIndex, setCopiedIndex] = useState(null);

  const content = [
    { heading: "delta modulation", text: `clc;
t=0:2*pi/100:2*pi;
x=5*sin(2*pi*t/5);
plot(x)
hold on
y=[0];
xr=0;
del=0.4;
for i=1:length(x)-1
if xr(i)<=x(i)
d=1;
xr(i+1)=xr(i)+del;
else
d=0;
xr(i+1)=xr(i)-del;
end
y=[y d];
end
stairs(xr);
hold off
MSE=sum((x-xr).^2)/length(x)
y`

},
    { heading: "Noise", text: `Fs = 1000;        % Sampling frequency in Hz
N = 1024;         % Number of samples
t = (0:N-1)/Fs;   % Time vector

% Generate Additive White Gaussian Noise (AWGN)
noise = randn(1, N);

% Plot Time-Domain Signal
figure;
plot(t, noise);
title('Additive White Gaussian Noise (Time Domain)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Compute and plot Power Spectral Density (PSD)
figure;
[psd, f] = pwelch(noise, [], [], [], Fs); % Welch's method for PSD estimation
plot(f, 10*log10(psd));
title('Power Spectral Density of AWGN');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% Compute and plot Autocorrelation in Frequency Domain
noise_fft = fft(noise);
auto_corr_freq = abs(noise_fft).^2;

figure;
plot(linspace(-Fs/2, Fs/2, N), fftshift(auto_corr_freq));
title('Autocorrelation in Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;
` },
    { heading: "ask psk fsk", text: `clc;
close all;
clear all;
n=10;
b=[1 0 0 1 1 1 0 1 0 1];
f1=1;
f2=2;
t=0:1/30:1-1/30;
%ASK
sa1=sin(2*pi*f1*t);
E1=sum(sa1.^2);
sa1=sa1/sqrt(E1);
sa0=0*sin(2*pi*f1*t);
%FSK
sf0=sin(2*pi*f1*t);
E0=sum(sf0.^2);
sf0=sf0/sqrt(E0);
sf1=sin(2*pi*f2*t);
E1=sum(sf1.^2);
sf1=sf1/sqrt(E1);
 
%PSK
sp=sin(2*pi*f1*t);
E1=sum(sp.^2);
sp0=-sin(2*pi*f1*t)/sqrt(E1);
sp1=sin(2*pi*f1*t)/sqrt(E1);
 
%MODULATION
ask=[];psk=[];fsk=[];
for i=1:n
    if b(i)==1
        ask=[ask sa1];
        psk=[psk sp1];
        fsk=[fsk sf1];
    else
        ask=[ask sa0];
        psk=[psk sp0];
        fsk=[fsk sf0];
    end
end
figure(1)
subplot(411)
stairs(0:10,[b(1:10) b(10)],'linewidth',1.5)
axis([0 10 -0.5 1.5])
title('Message Bits'); grid on
xlabel('Time');
ylabel('Amplitude')
 
subplot(412)
tb=0:1/30:10-1/30;
plot(tb, ask(1:10*30),'b','linewidth',1.5)
title('ASK Modulation'); grid on
xlabel('Time');
ylabel('Amplitude')
 
subplot(413)
plot(tb, fsk(1:10*30),'r','linewidth',1.5)
title('FSK Modulation'); grid on
xlabel('Time');
ylabel('Amplitude')
 
subplot(414)
plot(tb, psk(1:10*30),'k','linewidth',1.5)
title('PSK Modulation'); grid on
xlabel('Time');
ylabel('Amplitude')
` },
    { heading: "modulation and demodulation", text: `clc;
close all;
clear all;
n=input('Enter n value for n-bit PCM system: ');
n1=input('Enter number of samples in a period: ');
L=2^n;
x=0:2*pi/n1:4*pi;
s=8*sin(x);
subplot(3,1,1);
plot(s);
title('Analog signal');
ylabel('Amplitude--->')
xlabel('Time--->');
subplot(3,1,2);
stem(s);
grid on;
title('Sampled signal');
ylabel('Amplitude--->');
xlabel('Time--->');
vmax=8;
vmin=-vmax;
del=(vmax-vmin)/L;
part=vmin:del:vmax;
code=vmin-(del/2):del:vmax+(del/2);
[ind,q]=quantiz(s,part,code);
l1=length(ind);
l2=length(q);
for i = 1:l1
    if(ind(i)~=0)
        ind(i)=ind(i)-1;
    end
    i+1;
end
for i=l2
    if(q(i)==vmin-(del/2))
        q(i)=vmin+(del/2);
    end
end
subplot(3,1,3);
stem(q);
grid on;
title('Quantized signal');
ylabel('Amplitude---->');
xlabel('Time--->');
figure
code=de2bi(ind,'left-msb');
k=1;
for i=1:l1
    for j=1:n
        coded(k)=code(i,j);
        j=j+1;
        k=k+1;
    end
    i=i+1;
end
subplot(2,1,1);
grid on;
stairs(coded);
axis([0 100 -2 3]);
title('Encoded signal');
ylabel ('Amplitude--->');
qunt=reshape(coded,n,length(coded)/n);
index=bi2de(qunt','left-msb');
q=del*index+vmin+(del/2);
subplot(2,1,2);
grid on;
plot(q);
title('Demodulated signal');
ylabel('Amplitude--->');
xlabel ('Time---->');

` },
  ];

  const handleCopy = (text:any, index:any) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000); // Reset the button text after 2 seconds
    });
  };

  return (
    <div>
      {content.map((item, index) => (
        <div key={index} className="w-3/5 border-2 h-20 flex items-center justify-between px-4">
          <h1>{item.heading}</h1>
          <button
            onClick={() => handleCopy(item.text, index)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            {copiedIndex === index ? "Copied" : "Copy"}
          </button>
        </div>
      ))}
      <a 
      href="/sqare.jpg" 
      download="square.jpg" 
      className="mt-4 inline-block px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
    >
      Download 1st Experiment Photo
    </a>
    </div>
  );
}
