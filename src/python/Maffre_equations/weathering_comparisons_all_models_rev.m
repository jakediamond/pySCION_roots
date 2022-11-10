%%%%%%%%%%%%%%%%%%%%%% Present day data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% load CRU data
load CRU_dataset

%%%% read topo
load NASA_topo.mat

%%%% read regridded runoff from fekek eet al., 2002
load fekeke_run_regrid.mat
run_data(run_data<0) = 0 ;
run_data(isnan(run_data)==1) = 0 ;

%%%% find land
land = tmp_avg > -999 ;
%%% remove water temps at -999
tmp_avg(tmp_avg < -100) = 15 ;

%%% remove water
tmp_avg = tmp_avg.*land ;
tmp_avg(tmp_avg == 0) = NaN ;
pre_avg(pre_avg < 0) = 0 ;
ocean = (1 - land);
ocean(ocean==1) = NaN ;
ocean_move = ocean + 1 ;

%%%% flip
pre_avg = pre_avg' ;
tmp_avg = tmp_avg' ;
topo = flipud(NASA_topo) ;
topo(topo==99999) = 0 ;


%%%%%%%%%%%%%%%%%%%%%% HadCM3 present day %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%% weathering calculation
T = tmp_avg + 273 ; %%% K
% Q = pre_avg ; %%% mm/yr
Q = run_data ; %%% mm/yr
height = topo ; %%% m

%%%% grid box area in km2
Re = 6371 ; %%% earth radius km
delta_lat = 0.5 ; %%% model resolution lat
delta_lon = 0.5 ; %%% model resolution lon
dist_lat =  2* pi * Re * delta_lat / 360 ;
dist_lon =  2* pi * Re * cosd(lat) * delta_lon/ 360 ;
gridarea_vec = dist_lon .* dist_lat ;
for n = 1:720 
    gridarea(:,n) = gridarea_vec ;
end


%%%%% erosion rates from slope from Maffre
lat_grid = lat.*ones(360,720) ;
lon_grid = lon'.*ones(360,720) ;
%%%%% gradient calculation
[ASPECT,SLOPE,dFdyNorth,dFdxEast] = gradientm(lat_grid,lon_grid,height) ;
%%%%% topographic slope
tslope = ( dFdyNorth.^2 + dFdxEast.^2 ).^0.5 ;
%%%%% pierre erosion calculation
k_erosion = 6.5e-4 ;
TC = tmp_avg ;
epsilon = k_erosion .* (Q.^0.31) .* tslope .* max(TC,2) ;
%%%%% check total tonnes of erosion - should be ~16Gt
erosion_per_gridbox = epsilon .* gridarea .* 1e6 ; %%% t/m2/yr * m2
erosion_tot = sum(sum(erosion_per_gridbox)) ;

%%%% Pierre params
Xm = 0.1 ;
K = 6e-5 ; 
kw = 1e-3 ;
Ea = 20 ; 
z = 10 ; 
sigplus1 = 0.9 ; 

%%%% fixed params
T0 = 286 ;
R = 8.31e-3 ;

%%%% T, Q and erosion dependencies
R_T = exp( ( Ea ./ (R.*T0) ) - ( Ea ./ (R.*T) ) ) ;
R_Q = 1 - exp( -1.*kw .* Q ) ;
R_reg = ( (z./epsilon).^sigplus1 ) ./ sigplus1 ;

%%%% West (2012) combined dependency model, *1e6 to get to t/km2/yr
CW = epsilon .* Xm .* ( 1 - exp( -1.* K .* R_Q .* R_T .* R_reg ) ) * 1e6; 

%%%% Global chemical weathering rate for each gridcell
CW_times_area = CW .* gridarea ;

%%%% remove nans
CW_times_area_nonan = CW_times_area ;
CW_times_area_nonan(isnan(CW_times_area_nonan)==1) = 0 ;

%%%% integrated chemical weathering rate
CW_lats = sum(CW_times_area_nonan,2 ) ;
CW_integrated = sum( sum( CW_times_area_nonan ) ) ;

%%%% calculate fraction of CW in high mid lats
CW_hilat = sum(CW_lats(1:100)) + sum(CW_lats(261:360)) ;

%%%% runoff at lat bands
run_times_area = run_data .* gridarea ;
run_lats = sum(run_times_area,2 ) ;

pre_times_area = pre_avg .* gridarea ;
pre_lats = sum(pre_times_area,2 ) ;




%%%%%%%%%%%%%%%%%%%%%% FOAM present day %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load INTERPSTACK_rev2021.mat

lat_FOAM = INTERPSTACK.lat ;
lon_FOAM = INTERPSTACK.lon ;
land_FOAM = INTERPSTACK.land(:,:,21) ;
nanland_FOAM = ones(40,48) ;
nanland_FOAM(land_FOAM==0) = NaN ;
nanland_FOAM(land_FOAM==1) = 1 ;

%%%% weathering calculation fields
T_FOAM = INTERPSTACK.Tair(:,:,8,21) + 273 ; %%% K
Q_FOAM = INTERPSTACK.runoff(:,:,8,21) ; %%% mm/yr
height_FOAM = INTERPSTACK.topo(:,:,21) ; %%% m
height_FOAM(isnan(height_FOAM)==1) = 0 ;

%%%% only have land
T_FOAM = T_FOAM.*land_FOAM ;
Q_FOAM = Q_FOAM.*land_FOAM ;
height_FOAM = height_FOAM.*land_FOAM ;

%%%% grid box area in km2
gridarea_FOAM = INTERPSTACK.aire(:,:,21) .* 1e6 ; 

%%%%% erosion rates from slope from Maffre
lat_grid_FOAM = INTERPSTACK.lat'.*ones(40,48) ;
lon_grid_FOAM = INTERPSTACK.lon.*ones(40,48) ;
%%%%% gradient calculation
[ASPECT_FOAM,SLOPE_FOAM,dFdyNorth_FOAM,dFdxEast_FOAM] = gradientm(lat_grid_FOAM,lon_grid_FOAM,height_FOAM) ;
%%%%% topographic slope
tslope_FOAM = ( dFdyNorth_FOAM.^2 + dFdxEast_FOAM.^2 ).^0.5 ;
%%%%% pierre erosion calculation
% k_erosion_FOAM = 3.3e-3 ;
k_erosion_FOAM = 6.5e-4 ; %%%% use normal value

TC_FOAM = T_FOAM - 273 ;
epsilon_FOAM = k_erosion_FOAM .* (Q_FOAM.^0.31) .* tslope_FOAM .* max(TC_FOAM,2) ;
%%%%% check total tonnes of erosion - should be ~16Gt
erosion_per_gridbox_FOAM = epsilon_FOAM .* gridarea_FOAM .* 1e6 ; %%% convert to t/km2 by multiplying both erosion and gridarea into km2
erosion_tot_FOAM = sum(sum(erosion_per_gridbox_FOAM)) ;

%%%% T, Q and erosion dependencies
R_T_FOAM = exp( ( Ea ./ (R.*T0) ) - ( Ea ./ (R.*T_FOAM) ) ) ;
R_Q_FOAM = 1 - exp( -1.*kw .* Q_FOAM ) ;
R_reg_FOAM = ( (z./epsilon_FOAM).^sigplus1 ) ./ sigplus1 ;

%%%% West (2012) combined dependency model, *1e6 to get to t/km2/yr
CW_FOAM = epsilon_FOAM .* Xm .* ( 1 - exp( -1.* K .* R_Q_FOAM .* R_T_FOAM .* R_reg_FOAM ) ) * 1e6; 

%%%% Global chemical weathering rate for each gridcell
CW_times_area_FOAM = CW_FOAM .* gridarea_FOAM ;

%%%% remove nans
CW_times_area_nonan_FOAM = CW_times_area_FOAM ;
CW_times_area_nonan_FOAM(isnan(CW_times_area_nonan_FOAM)==1) = 0 ;

%%%% integrated chemical weathering rate
CW_lats_FOAM = sum(CW_times_area_nonan_FOAM,2 ) ;
CW_integrated_FOAM = sum( sum( CW_times_area_nonan_FOAM ) ) ;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%     world rivers comparison   %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% from present day climatology

%%%% load world river data
polygon_area = ncread('all_basin_0.5deg.nc','polygon_area');

%%%% make ploygon map for plot
totalpolygons = polygon_area(:,:,1) ;
for n=2:80
    totalpolygons = totalpolygons + polygon_area(:,:,n) ;
end


%%%% compute world rivers weathering from present day data
for n=1:80
    thisarea = (polygon_area(:,:,n))./1000000;
    basinweathering = CW.*thisarea';
    basinweathering(isnan(basinweathering)) = 0;
    bulkbasinweathering(n) = sum(sum(basinweathering));
end


%%%% from FOAM
polygon_area_FOAM = ncread('all_basin_FOAM-48x40.nc','polygon_area');

%%%% make ploygon map for plot
totalpolygons_FOAM = polygon_area_FOAM(:,:,1) ;
for n=2:80
    totalpolygons_FOAM = totalpolygons_FOAM + polygon_area_FOAM(:,:,n) ;
end


%%%% compute world rivers weathering from present day data
for n=1:80
    thisarea_FOAM = (polygon_area_FOAM(:,:,n))./1000000;
    basinweathering_FOAM = CW_FOAM.*thisarea_FOAM';
    basinweathering_FOAM(isnan(basinweathering_FOAM)) = 0;
    bulkbasinweathering_FOAM(n) = sum(sum(basinweathering_FOAM));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%     plotting script   %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%% plotting colours

%%%% IPCC precip
IPCC_pre = [ 246 232 195 ;
245 245 245 ;
199 234 229 ;
128 205 193 ;
53 151 143 ;
1 102 94 ;
0 60 48 ] ./ 255 ;

%%%% IPCC temp colorbar
IPCC_temp = flipud( [103 0 31 ;
178 24 43 ;
214 96 77 ;
244 165 130 ;
253 219 199 ;
247 247 247 ;
209 229 240 ;
146 197 222 ;
67 147 195 ;
33 102 172 ;
5 48 97 ]./ 255 ) ;

%%%%% sequential colorscheme
new_seq = [ 70, 49, 70 ;
86, 72, 101 ;
96, 99, 133 ;
99, 127, 165 ;
94, 158, 195 ;
82, 189, 220 ]./ 255 ;

%%%% plot
figure

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Plot present day data %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% temp
subplot(2,6,1)
h = pcolor(lon,lat,tmp_avg(:,:)) ;
set(h,'edgecolor','none')
box on
colormap( gca, IPCC_temp)
colorbar
%%%% precip
subplot(2,6,2)
h = pcolor(lon,lat,run_data(:,:).*ocean_move') ;
set(h,'edgecolor','none')
box on
colormap( gca, IPCC_pre)
caxis([0 1000])
colorbar
%%%% erosion in t/m2/yr
subplot(2,6,3)
h = pcolor(lon,lat,1e6.*epsilon(:,:).*ocean_move') ;
set(h,'edgecolor','none')
box on
colormap( gca, 'pink')
colorbar
caxis([0 1000])
%%%% weathering
subplot(2,6,4)
h = pcolor(lon,lat,CW(:,:)) ;
set(h,'edgecolor','none')
box on
colormap( gca, new_seq)
caxis([0 10])
colorbar

%%%% latitudenal CW and runoff
subplot(2,6,5)
plot(CW_lats,lat)
xlabel('CW')
ylabel('lat')
ylim([-90 90])
grid on

%%%%% world rivers comparison
load WorldRivers.mat
% plot present comparison
subplot(2,6,6)
loglog(WorldRivers.*1e6,bulkbasinweathering,'x','MarkerSize',10)
xlim([1E3 1E8])
ylim([1E3 1E8])
xlabel('Data')
ylabel('Model')
hold on
% plot equivalence line
plot([1E3 1E8],[1E3 1E8],'r')
% plot 10 fold difference
plot([1E4 1E8],[1E3 1E7],'c')
plot([1E3 1E7],[1E4 1E8],'c')
% plot 5 fold difference
plot([5E3 1E8],[1E3 2E7],'b')
plot([1E3 2E7],[5E3 1E8],'b')
% plot 2 fold difference
plot([2E3 1E8],[1E3 6E7],'m')
plot([1E3 6E7],[2E3 1E8],'m')


% %%%% check statistical fit
% fitlm(bulkbasinweathering,WorldRivers.*1e6)
% grid on


%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Plot FOAM %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% temp
subplot(2,6,7)
h = pcolor(lon_FOAM,lat_FOAM,T_FOAM(:,:).*nanland_FOAM) ;
set(h,'edgecolor','none')
box on
colormap( gca, IPCC_temp)
colorbar
%%%% precip
subplot(2,6,8)
h = pcolor(lon_FOAM,lat_FOAM,Q_FOAM.*nanland_FOAM) ;
set(h,'edgecolor','none')
box on
colormap( gca, IPCC_pre)
caxis([0 1000])
colorbar
%%%% erosion in t/m2/yr
subplot(2,6,9)
h = pcolor(lon_FOAM,lat_FOAM,1e6.*epsilon_FOAM.*nanland_FOAM) ;
set(h,'edgecolor','none')
box on
colormap( gca, 'pink')
colorbar
caxis([0 1000])
%%%% weathering
subplot(2,6,10)
h = pcolor(lon_FOAM,lat_FOAM,CW_FOAM(:,:)) ;
set(h,'edgecolor','none')
box on
colormap( gca, new_seq)
caxis([0 10])
colorbar

%%%% latitudenal CW and runoff
subplot(2,6,11)
plot(CW_lats_FOAM,lat_FOAM)
xlabel('CW')
ylabel('lat')
ylim([-90 90])
grid on

%%%%% world rivers comparison
% plot FOAM comparison
subplot(2,6,12)
loglog(WorldRivers.*1e6,bulkbasinweathering_FOAM,'x','MarkerSize',10)
xlim([1E3 1E8])
ylim([1E3 1E8])
xlabel('Data')
ylabel('Model')
hold on
% plot equivalence line
plot([1E3 1E8],[1E3 1E8],'r')
% plot 10 fold difference
plot([1E4 1E8],[1E3 1E7],'c')
plot([1E3 1E7],[1E4 1E8],'c')
% plot 5 fold difference
plot([5E3 1E8],[1E3 2E7],'b')
plot([1E3 2E7],[5E3 1E8],'b')
% plot 2 fold difference
plot([2E3 1E8],[1E3 6E7],'m')
plot([1E3 6E7],[2E3 1E8],'m')
