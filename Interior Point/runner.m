n_vals = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
time_mean_list = zeros(11,1);
time_ci_lower_list = zeros(11,1);
time_ci_upper_list = zeros(11,1);
fx_mean_list = zeros(11,1);
fx_ci_lower_list = zeros(11,1);
fx_ci_upper_list = zeros(11,1);
xnorms_mean_list = zeros(11,1);
xnorms_ci_lower_list = zeros(11,1);
xnorms_ci_upper_list = zeros(11,1);
random_iterations_number = 5;
k = 3;
for i=1:11
        n = n_vals(i);
        [time_n_k, f_x, xs, xs_norms] = evaluateInteriorPoint(n, k, random_iterations_number);
        zstar = 1.96;
        time_mean = mean(time_n_k);
        time_std = std(time_n_k);
        time_ci_lower = time_mean - (zstar * time_std/(sqrt(random_iterations_number)));
        time_ci_upper = time_mean + (zstar * time_std/(sqrt(random_iterations_number)));
        time_mean_list(i) = (time_mean);
        time_ci_lower_list(i) = (time_ci_lower);
        time_ci_upper_list(i) = (time_ci_upper);

        fx_mean = mean(f_x);
        fx_std = std(f_x);
        fx_ci_lower = fx_mean - (zstar * fx_std/(sqrt(random_iterations_number)));
        fx_ci_upper = fx_mean + (zstar * fx_std/(sqrt(random_iterations_number)));

        fx_mean_list(i) = (fx_mean);
        fx_ci_lower_list(i) = (fx_ci_lower);
        fx_ci_upper_list(i) = (fx_ci_upper);

        xnorms_mean = mean(xs_norms);
        xnorms_std = std(xs_norms);
        xnorms_ci_lower = xnorms_mean - (zstar * xnorms_std/(sqrt(random_iterations_number)));
        xnorms_ci_upper = xnorms_mean + (zstar * xnorms_std/(sqrt(random_iterations_number)));

        xnorms_mean_list(i) = (xnorms_mean);
        xnorms_ci_lower_list(i) = (xnorms_ci_lower);
        xnorms_ci_upper_list(i) = (xnorms_ci_upper);

        xsshaped = reshape(xs, [k, n]);
        fileID = fopen(strcat('plot_k',num2str(n),'.txt'),'w');
        for j = 1:n
            fprintf(fileID,'%f %f %f\n',xsshaped(1,j), xsshaped(2,j),xsshaped(3,j));
            %plot(xsshaped(1,j), xsshaped(2,j), 'o')
            %hold on
        end
        %xlim([-2 2])
        %ylim([-2 2])
        %saveas(gcf,strcat(num2str(n),'.png'));
        %close
end
plot(n_vals, fx_mean_list);   
xlabel('n')
ylabel('fx') 
saveas(gcf, 'fx.png')

close
plot(n_vals, time_mean_list);   
xlabel('n')
ylabel('Time(Seconds)') 
saveas(gcf, 'time.png')

close
plot(n_vals, xnorms_mean_list);   
xlabel('n')
ylabel('Infeasibility Term') 
saveas(gcf, 'norms.png')
close
fileID = fopen(strcat('projected_k',num2str(n),'.txt'),'w');
fprintf(fileID,'fx\n');
for i=1:11
    n = n_vals(i);
    fprintf(fileID,"%d\t",n);
end
fprintf(fileID,'\n');

fprintf(fileID,'n\n');    
for i=1:11
    fprintf(fileID,"%f\t",fx_mean_list(i));
end
fprintf(fileID,'\n');
for i=1:11
    fprintf(fileID,"%f\t", fx_ci_lower_list(i));
end
fprintf(fileID,'\n');
for i=1:11
    fprintf(fileID,"%f\t", fx_ci_upper_list(i));
end
fprintf(fileID,'\n');

fprintf(fileID,'Norms\n');
for i=1:11
    fprintf(fileID,'%.17f\t',xnorms_mean_list(i));
end
fprintf(fileID,'\n');
for i=1:11
    fprintf(fileID,'%.17f\t',xnorms_ci_lower_list(i));
end
fprintf(fileID,'\n');
for i=1:11
    fprintf(fileID,'%.17f\t',xnorms_ci_upper_list(i));
end
fprintf(fileID,'\n');

fprintf(fileID,'Time(sec)\n');
for i=1:11
    fprintf(fileID,'%f\t',time_mean_list(i));
end
fprintf(fileID,'\n');

for i=1:11
    fprintf(fileID,'%f\t',time_ci_lower_list(i));
end
fprintf(fileID,'\n');

for i=1:11
    fprintf(fileID,'%f\t',time_ci_lower_list(i));
end
fprintf(fileID,'\n');