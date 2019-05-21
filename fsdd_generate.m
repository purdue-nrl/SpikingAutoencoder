% the file is to process audio data

addpath(genpath('../free-spoken-digit-dataset/recordings'));
%addpath(genpath('../AuditoryToolbox/'));

window = 64;
noverlap = 0;
Fs = 8000;
nfft = 256;
[y0, Fs0] = audioread('2_jackson_0.wav');
[y1, Fs1] = audioread('2_jackson_1.wav');
[y2, Fs2] = audioread('2_jackson_2.wav');
[y3, Fs3] = audioread('2_jackson_3.wav');
[s0, f0, t0] = spectrogram(y0, window);
[s1, f1, t1] = spectrogram(y1, window);
[s2, f2, t2] = spectrogram(y2, window);
[s3, f3, t3] = spectrogram(y3, window);

figure(1);
colormap default;
subplot(2,2,1);
imagesc(abs(s0));
subplot(2,2,2);
imagesc(abs(s1));
subplot(2,2,3);
imagesc(abs(s2));
subplot(2,2,4);
imagesc(abs(s3));

% [y0, Fs0] = audioread('2_jackson_0.wav');
% s0 = LyonPassiveEar(y0,Fs0, 16, 8, 8/32, 0, 1, 3);
% figure(1);
% imagesc(s0);


