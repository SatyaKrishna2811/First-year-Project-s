close all;                                                       
clear ;                                                          
clc;                                                               

ref_size=0:0.01:100;                   %reference size matrix%
%Matrix
T= tril(ones(length(ref_size)));       % Lower triangular matrix of the reference size matrix                                          


%Defining constants
r=0.529*10^-10;                        %radius of the hydrogen atom                              
%v=5;                                                                      
v=2.18*10^6;                           %velocity of the electron                                    
w=v/r;                                 %angular velocity of the electron                                     
W=w*(ones(length(ref_size),1));        %size of the matrix of angular velocity


theta=T*W;                            %angulaqr displacement
x=cos(theta');                        %x cordinate for the circular motion 
y=sin(theta');                        %y cordinate for the circular motion

xy=[x ; y];                           %combination of x,y into a matrix
R=[r 0;0 r];                          %orbit radius matrix

Path=R*xy;                            %path of the electron

plot3(0,0,0,'r.','MarkerSize',20);    %ploting the nucleus with the red dot of size 20
hold on;
Electron=plot3(r,0,0,'k.','MarkerSize',20); %ploting the electron with the black of size 20
hold on;
grid on;                                    %to plot it in the square boxes


xlim([-1.5*r,1.5*r]);                       %ploting the x_axix
ylim([-1.5*r,1.5*r]);                       %ploting the y_axis
zlim([-1.5*r,1.5*r]);                       %ploting the z_axis
for i=1:length(ref_size)
    Electron.XData=Path(1,i);
    Electron.YData=Path(2,i);
    Electron.ZData=0;
    plot3(Path(1,i),Path(2,i),0,'b.','MarkerSize',0.001)
    pause(0.1)
end


