clear; close all; clc;

%% Load BP Data:
bpcsv = importdata('BPData.csv');


%% Load Data:

matFiles = dir('ProcessedDataNoVideo/*.mat');
allFeatures = [];
plot_on = 0;
for mf = 1:length(matFiles)
    mf
    data = load(fullfile(matFiles(mf).folder,matFiles(mf).name));
    
    pid = matFiles(mf).name(1:end-5);
    f = find(ismember(bpcsv.textdata,pid));
    bpsys(mf) = bpcsv.data(f,4);
    bpdia(mf) = bpcsv.data(f,8);
    hr(mf) = bpcsv.data(f,12);
    age(mf) = bpcsv.data(f,13);
    wgt(mf) = bpcsv.data(f,14);
    hgt(mf) = bpcsv.data(f,15);

    data.ppg = 1-data.ppg;

    %% Plot Raw Green Channel:


    figure('Renderer', 'painters', 'Position', [10 10 1500 2000])
    subplot(6,2,1); plot(data.ppg_face); title('Raw Average Green Channel');
    subplot(6,2,2); plot(data.ppg); title('Finger PPG');

    %% Remove artifacts:
    t = [1:length(data.ppg_face)]; finger_t = [1:length(data.ppg)];
    ppgmean = mean(data.ppg_face);
    ppgstdev = std(data.ppg_face(1:200));
    f1 = find(data.ppg_face > ppgmean+3*ppgstdev);
    f2 = find(data.ppg_face < ppgmean-3*ppgstdev);
    f = [f1 f2];
    %f = intersect(f1, f2);
    %t = t(f);
    %data.ppg_face = data.ppg_face(f);
    data.ppg_face(f) = 0;
    subplot(6,2,3); plot(t, data.ppg_face); title('Raw Average Green Channel - Artifacts Removed');

    %% Filter:
    LPF = 0.7; %low cutoff frequency (Hz) - 0.7 Hz in reference
    HPF = 2; %high cutoff frequency (Hz) - 4.0 Hz in reference
    FS = 60;
    NyquistF = 1/2*FS;
    [B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%Butterworth 3rd order filter
    data.ppg_face_filt = filtfilt(B,A,double(data.ppg_face));
    
    HPF = 8; %high cutoff frequency (Hz) - 4.0 Hz in reference 
    [B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%Butterworth 3rd order filter
    data.ppg_face_filt_4 = filtfilt(B,A,double(data.ppg_face));
    
    mPxx = 0;
    for i=1:5
        bvp = data.ppg_face_filt_4((i-1)*1300+1:i*1300);
        [PR,Pxx,F] = prpsd(bvp,FS,40,240,false);
        freqIdx = (F>0.7 & F<2);
        %figure; subplot(2,1,1); plot(bvp); subplot(2,1,2); plot(F(freqIdx),Pxx(freqIdx));
        Pxx=Pxx/sum(Pxx);
        if max(Pxx)>mPxx,
            mPxx = max(Pxx);
            idx = i;
        end
    end
    
    %t = t((idx-1)*1300+1:idx*1300);
    %data.ppg_face_filt = data.ppg_face_filt((idx-1)*1300+1:idx*1300);
    %data.ppg_face_filt_4 = data.ppg_face_filt_4((idx-1)*1300+1:idx*1300);
    subplot(6,2,5); plot(t, data.ppg_face_filt);  title('Filtered');

    %% Detect Peaks:
    data.ppg_face_filt = data.ppg_face_filt-min(data.ppg_face_filt);
    data.ppg_face_filt = data.ppg_face_filt/max(data.ppg_face_filt);
    [peaks, peaks_i] = findpeaks(data.ppg_face_filt, 'MinPeakHeight', 0.7, 'MinPeakDistance', 20);
    subplot(6,2,7); plot(t, data.ppg_face_filt);
    hold on
    plot(t(peaks_i), data.ppg_face_filt(peaks_i),'ro'); 

    %% Detect Troughs:
    [troughs, troughs_i] = findpeaks(1-data.ppg_face_filt, 'MinPeakHeight', 0.7, 'MinPeakDistance', 20);
    hold on
    plot(t(troughs_i), data.ppg_face_filt(troughs_i),'go'); title('Raw Average Green Channel - Peaks and Troughs Detected');


    data.ppg = data.ppg-min(data.ppg);
    data.ppg = data.ppg/max(data.ppg);
    [finger_peaks, finger_peaks_i] = findpeaks(data.ppg, 'MinPeakHeight', 0.7, 'MinPeakDistance', 20);
    subplot(6,2,8); plot(finger_t, data.ppg);
    hold on
    plot(finger_t(finger_peaks_i), data.ppg(finger_peaks_i),'ro');

    [finger_troughs, finger_troughs_i] = findpeaks(1-data.ppg, 'MinPeakHeight', 0.7, 'MinPeakDistance', 20);
    hold on
    plot(finger_t(finger_troughs_i), data.ppg(finger_troughs_i),'go'); title('Raw Average Green Channel - Peaks and Troughs Detected');



    %% Remove Ectopic Beats:
    fs = 60;
    RR = diff(peaks_i);
    Rpos = peaks_i;
    [Rpos, RR, Pod] = ectopic_detection_correction(fs,RR,Rpos);

    %% Plot IBIs:
    subplot(6,2,9); plot(Rpos(2:end), RR/60);
    subplot(6,2,10); plot(finger_peaks_i(2:end), diff(finger_peaks_i)/60);


    f = [];

    %% Pulse Segmentation:
    tdia = 0;
    
    tmpFeatures = [];
    try, 
        for nP = 3:50,
            %nP = 1; % Peak number.
            if (peaks_i(nP+1)-peaks_i(nP))/fs>1.5 || (peaks_i(nP+1)-peaks_i(nP))/fs<0.5,
            %    continue
            end
            ppg = 1-data.ppg_face_filt_4(peaks_i(nP):peaks_i(nP+1));

            subplot(6,2,11);   plot(diff(ppg));  hold on;

            ppgFirstDeriv = diff(ppg);
            if plot_on,
                subplot(3,1,2); plot(diff(ppg));
            end

            ppgSecondDeriv = diff(ppgFirstDeriv);
            [pks,tpks] = findpeaks(ppgSecondDeriv);

            if plot_on,
                subplot(3,1,3); plot(ppgSecondDeriv);
                hold on; plot(tpks,pks,'ro')
            end
            
            [Psys, tsys] = max(ppg);

            % f1 => (tsys-tdia)/fs
            f(1) = (tsys-tdia)/fs;

            if plot_on,
                subplot(3,1,1); plot([tsys tsys], [min(ppg) max(ppg)]);
                subplot(3,1,1); plot(tsys,Psys,'ro');
            end

            idx = tpks>tsys;
            [m,i] = max(idx);
            Pd = ppg(tpks(i));
            td = tpks(i);

            if plot_on,
                subplot(3,1,1); plot([td td], [min(ppg) max(ppg)])
                subplot(3,1,1); plot(td,Pd,'ro')
            end

            % f2 => td-tdia
            f(2) = (td-tdia)/fs;

            % f3 => (td-tdia)/RR
            f(3) = f(2)/(length(ppg)/fs);

            % f4 => (Pd-Pdia)/|Psys-Pdia|
            Pdia = ppg(tdia+1);
            f(4) = (Pd-Pdia)/abs(Psys-Pdia);

            % f5 => max(dP/dt)/|Psys-Pdia|
            [m,i] = max(ppgFirstDeriv);
            f(5) = m/abs(Psys-Pdia);

            % f6 => RR
            f(6) = length(ppg)/fs;

            % f7 => tdia - td
            f(7) = length(ppg)/fs - td/fs;

            tmpFeatures = [tmpFeatures; f];
        end
        allFeatures = [allFeatures; median(tmpFeatures)];
    catch
        allFeatures = [allFeatures; zeros(1,7)];
    end
    
    close all

end

featureNames = {'Sys. Ramp','Eject Dur.','Eject Dur./RR','Norm. Dia. Notch Hei.','Max Sys. Ramp','RR','Dia. Notch to Foot','Sys. Peak to Foot'}
for i=1:7
    f = find(allFeatures(:,i)~=0);
    subplot(3,7,i); scatter(allFeatures(f,i), bpsys(f)); axis square;
    [r,p] = corr(allFeatures(f,i), bpsys(f)');
    title([featureNames{i}, ': r: ',num2str(r,'%4.2f'),', p: ',num2str(p,'%4.2f')]);
    if i==1, ylabel('BP'); end
end
allFeaturesNorm = allFeatures./hr'
for i=1:7
    f = find(allFeaturesNorm(:,i)~=0);
    subplot(3,7,i+7); scatter(allFeaturesNorm(f,i), bpsys(f)); axis square;
    [r,p] = corr(allFeaturesNorm(f,i), bpsys(f)');
    title([featureNames{i}, ': r: ',num2str(r,'%4.2f'),', p: ',num2str(p,'%4.2f')]);
    if i==1, ylabel('BP'); end
end
subplot(3,7,15); scatter(age, bpsys); axis square;
[r,p] = corr(age', bpsys');
title(['AGE: r: ',num2str(r,'%4.2f'),', p: ',num2str(p,'%4.2f')]);
ylabel('BP'); 
subplot(3,7,16); scatter(wgt, bpsys); axis square;
[r,p] = corr(wgt', bpsys');
title(['WEIGHT: r: ',num2str(r,'%4.2f'),', p: ',num2str(p,'%4.2f')]);
subplot(3,7,17); scatter(hgt, bpsys); axis square;
[r,p] = corr(hgt', bpsys');
title(['HEIGHT: r: ',num2str(r,'%4.2f'),', p: ',num2str(p,'%4.2f')]);
subplot(3,7,18); scatter(wgt./hgt, bpsys); axis square;
[r,p] = corr((wgt./hgt)', bpsys');
title(['WEIGHT/HEIGHT: r: ',num2str(r,'%4.2f'),', p: ',num2str(p,'%4.2f')]);
subplot(3,7,19); scatter(hr, bpsys); axis square;
[r,p] = corr((hr)', bpsys');
title(['HR: r: ',num2str(r,'%4.2f'),', p: ',num2str(p,'%4.2f')]);


%    f = find(allFeatures(:,i)~=0);
%    scatter(mean(allFeatures(f,1:3)')', bpsys(f)); axis square;
%    [r,p] = corr(mean(allFeatures(f,1:3)')', bpsys(f)');
%    title(['Corr: ',num2str(r),', p: ',num2str(p)])

keyboard



%     %% Pulse Segmentation:
%     figure;
% 
%     for i=1:25
%         subplot(5,5,i); plot(1-data.ppg_face_filt(peaks_i(i):peaks_i(i+1)));
%         hold on
%         plot(1-data.ppg(finger_peaks_i(i):finger_peaks_i(i+1)));
%     end



plot_on = 0;
if plot_on, figure; end

f = []
allFeatures = [];
for i=1:length()
    tdia = 0;
    
    %plot(1-data.ppg_face_filt(peaks_i(i):peaks_i(i+1)));
    %hold on
    %ppg = 1-data.ppg(finger_peaks_i(i):finger_peaks_i(i+1));
    ppg = 1-data.ppg_face_filt(peaks_i(i):peaks_i(i+1));
    
    if plot_on,
        subplot(3,1,1); plot(ppg);
        hold on
    end
    
    ppgFirstDeriv = diff(ppg);
    subplot(3,1,2); plot(diff(ppg));
    
    ppgSecondDeriv = diff(ppgFirstDeriv);
    [pks,tpks] = findpeaks(ppgSecondDeriv);
    
    if plot_on,
        subplot(3,1,3); plot(ppgSecondDeriv);
        hold on; plot(tpks,pks,'ro')
    end
    
    [Psys, tsys] = max(ppg);
    
    % f1 => tsys-tdia
    f(1) = (tsys-tdia)/fs;
    
    if plot_on,
        subplot(3,1,1); plot([tsys tsys], [min(ppg) max(ppg)]);
        subplot(3,1,1); plot(tsys,Psys,'ro');
    end
    
    idx = tpks>tsys;
    [m,i] = max(idx);
    Pd = ppg(tpks(i));
    td = tpks(i);
    
    if plot_on,
        subplot(3,1,1); plot([td td], [min(ppg) max(ppg)])
        subplot(3,1,1); plot(td,Pd,'ro')
    end
    
    % f2 => td-tdia
    f(2) = (td-tdia)/fs;
    
    % f3 => (td-tdia)/RR
    f(3) = f(2)/(length(ppg)/fs);
    
    % f4 => (Pd-Pdia)/|Psys-Pdia|
    Pdia = ppg(tdia+1);
    f(4) = (Pd-Pdia)/abs(Psys-Pdia);
    
    % f5 => max(dP/dt)/|Psys-Pdia|
    [m,i] = max(ppgFirstDeriv);
    f(5) = m/abs(Psys-Pdia);
    
    % f6 => RR
    f(6) = length(ppg)/fs;
    
    % f7 => tdia - td
    f(7) = length(ppg)/fs - td/fs;
    
    allFeatures = [allFeatures; f];
    close all
end

