disp(' '); disp('clear all, close all')
clear all, close all
write_fig = 0;
disp(' ')

disp('n1 = 80; n2 = 40;                   % number of data points from each class')
n1 = 80; n2 = 40;
disp('S1 = eye(2); S2 = [1 0.95; 0.95 1];           % the two covariance matrices')
S1 = eye(2); S2 = [1 0.95; 0.95 1];
disp('m1 = [0.75; 0]; m2 = [-0.75; 0];                            % the two means')
m1 = 0.75; m2 = -0.75;
disp(' ')

disp('x1 = bsxfun(@plus, chol(S1)''*gpml_randn(0.2, 2, n1), m1);')
x1 = bsxfun(@plus, gpml_randn(0.2, 1, n1), m1);
disp('x2 = bsxfun(@plus, chol(S2)''*gpml_randn(0.3, 2, n2), m2);')         
x2 = bsxfun(@plus, gpml_randn(0.3, 1, n2), m2);         
disp(' ')

x1
x2

% Combined x1 and x2
xC = [x1 x2];

disp('x = [x1 x2]''; y = [-ones(1,n1) ones(1,n2)]'';')
x = [x1 x2]'; y = [sign((x1-2).^3 - 5*(x1-2).^2 + 8) sign((x2-2).^3 - 5*(x2-2).^2 + 8)]';
x
figure(6)
disp('plot(x1(1,:), x1(2,:), ''b+''); hold on;');
plot(xC(sign((xC-2).^3 - 5*(xC-2).^2 + 8) > -1), 1, 'b+', 'MarkerSize', 12); hold on
disp('plot(x2(1,:), x2(2,:), ''r+'');');
plot(xC(sign((xC-2).^3 - 5*(xC-2).^2 + 8) == -1), -1, 'r+', 'MarkerSize', 12);
disp(' ')


% disp('[t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);')
% [t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);
% disp('t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs')
% t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs
t1 = [-4:0.1:4,-4:0.1:4]
t = [t1(:)]; n = length(t);
% disp('tmm = bsxfun(@minus, t, m1'');')
% tmm = bsxfun(@minus, t, m1');
% disp('p1 = n1*exp(-sum(tmm*inv(S1).*tmm/2,2))/sqrt(det(S1));')
% p1 = n1*exp(-sum(tmm*inv(S1).*tmm/2,2))/sqrt(det(S1));
% disp('tmm = bsxfun(@minus, t, m2'');')
% tmm = bsxfun(@minus, t, m2');
% disp('p2 = n2*exp(-sum(tmm*inv(S2).*tmm/2,2))/sqrt(det(S2));')
% p2 = n2*exp(-sum(tmm*inv(S2).*tmm/2,2))/sqrt(det(S2));
% set(gca, 'FontSize', 24)
% disp('contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.1:0.1:0.9])')
% contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.1:0.1:0.9])
% [c h] = contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.5 0.5]);
% set(h, 'LineWidth', 2)
colorbar
grid
axis([-4 4 -4 4])
if write_fig, print -depsc f6.eps; end
% disp(' '); disp('Hit any key to continue...'); pause

disp(' ')
disp('meanfunc = @meanConst; hyp.mean = 0;')
meanfunc = @meanConst; hyp.mean = 0;
disp('covfunc = @covSEard;   hyp.cov = log([1 1 1]);')
covfunc = @covSEard;   hyp.cov = log([1 1]);
disp('likfunc = @likErf;')
likfunc = @likErf;
disp(' ')

disp('hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);')
hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);
disp('[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));')
[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n,1));
disp(' ')
figure(7)
set(gca, 'FontSize', 24)

% disp('contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)')
% contour(t1, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
% [c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
% set(h, 'LineWidth', 2)
% colorbar
grid
axis([-4 4 -4 4])
if write_fig, print -depsc f7.eps; end



z = linspace(-4.0, 4.0, 81)';
disp('[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);')
[m s2] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, z);

set(gca, 'FontSize', 24)
disp(' ')
disp('f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];') 
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
disp('fill([z; flipdim(z,1)], f, [7 7 7]/8);')
fill([z; flipdim(z,1)], f, [7 7 7]/8);

disp('hold on; plot(z, m); plot(x, y, ''+'')')
hold on; plot(z, m, 'LineWidth', 2)%; plot(x, y, '+', 'MarkerSize', 12)
% axis([-1.9 1.9 -0.9 3.9])
grid on
xlabel('input, x')
ylabel('output, y')

disp('plot(x1(1,:), x1(2,:), ''b+''); hold on')
plot(xC(sign((xC-2).^3 - 5*(xC-2).^2 + 8) > -1), 1, 'b+', 'MarkerSize', 12); hold on
disp('plot(x2(1,:), x2(2,:), ''r+'')')
plot(xC(sign((xC-2).^3 - 5*(xC-2).^2 + 8) == -1), -1, 'r+', 'MarkerSize', 12)

grid
axis([-4 4 -4 4])
if write_fig, print -depsc f2.eps; end
disp(' '); disp('Hit any key to continue...'); pause