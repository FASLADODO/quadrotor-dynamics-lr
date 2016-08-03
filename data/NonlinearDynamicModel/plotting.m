clear all;
close all;

 encoded = dlmread('speedDataNormalized.csv');
 decoded = dlmread('decodedSpeeds2.csv');
 
 figure;
 
 A = encoded';
 B= decoded';
 
 plot(A(:));
 hold on;
 plot(B(:));
 
 
 
 
 