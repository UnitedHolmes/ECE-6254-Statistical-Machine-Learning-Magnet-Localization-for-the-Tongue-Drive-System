clear
close all
clc

%folder_name='C:\code\MagneticLocalization\MagneticLocalization\data\processed_data\robotic_5mm';
%b/c robotic_5mm has 17 columns while
%"processed_data\prop2d_7cm_new\input.m" has 16 columns: input.m does
%not have Z coordinate,
folder_name='C:\Holmes\College study\Graduate Studies\2017 Spring\ECE 6254\Final Project\MagneticLocalization-master\6254 Progress\Yuanda';

input_file_name = strcat(folder_name,'\input_new.mat');

load(input_file_name)
input_original=input;

%the simulated input.m already has the z-coordinate added, and X,Y,Z coordinate in mm scale 


%% info to add if loading "...processed_data\prop2d_7cm_new\input.m" has 16 columns: input.m does not have Z coordinate.

% %add a Z coordinate =0 to the input dataset
% input=[input_original(:,1:14),zeros(size(input,1),1),input_original(:,15:16)];

% %convert X,Y,Z coordinate from cm to mm scale
% input(:,13:15)=input(:,13:15)*10;


%% test

test=input(input(:,16)>=225*7 & mod(input(:,13),10)==0 &mod(input(:,14),10)==0 ,:);
output_file_name=strcat(folder_name,'\test\test1.mat');
save(output_file_name,'test');
clearvars test
 
test=input(input(:,16)>=225*7,:);
output_file_name=strcat(folder_name,'\test\test2.mat');
save(output_file_name,'test');
clearvars test
 
test=input(input(:,16)>=225*7 & abs(input(:,13))<=30 & abs(input(:,14)-10)<=30,:);
output_file_name=strcat(folder_name,'\test\test3.mat');
save(output_file_name,'test');
clearvars test
 
test=input(input(:,16)>=225*7 & abs(input(:,13))<=25 & abs(input(:,14)-10)<=25,:);
output_file_name=strcat(folder_name,'\test\test4.mat');
save(output_file_name,'test');
clearvars test
 
test=input(input(:,16)>=225*7 & abs(input(:,13))<=20 & abs(input(:,14)-10)<=20,:);
output_file_name=strcat(folder_name,'\test\test5.mat');
save(output_file_name,'test');
clearvars test

%% train, cross validation

% extract the first num_point points for trial 1 to num_trial

num_point=10:10:80;

num_trial=1:7;

for m=1:length(num_point)
    for n=1:length(num_trial)

        for i=1:225*num_trial(n)
            temp1=input(input(:,16)==i-1,:);
            temp2((i-1)*num_point(m)+1:i*num_point(m),:)=temp1(1:num_point(m),:);
        end

        tcv=temp2( (mod(temp2(:,13),10)==0 & mod(temp2(:,14),10)==0) ,:);
        
        temp_name=strcat(folder_name,'\tcv\tcv_',num2str(num_trial(n)),'_',num2str(num_point(m)),'.mat');
        
        save(temp_name,'tcv')
        clearvars -except input num_point num_trial m n folder_name
    end
end


%% Display some figure
% figure;
% plot(input(input(:,15)==0,1));
% title_name=sprintf('(-3.5,-2.5), Trial 1, sensor 1, x axis, preprocess');
% title(title_name,'FontSize',14);
% xlabel('samples','FontSize',14);
% ylabel('Magnetic Flux (per 0.479 mgauss)','FontSize',14)
% 
% saveas(gcf,'../figures/preprocessdata2.png');