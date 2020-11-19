# dreamworld1
<Dream World 1, Volume Cloud>
Free-License

[Contribution guidelines for this project](asset_result/temp.png)


Teams:
Probability: { 0 <= x <= 1 }

1. Lighting 
  1.1 Transmittance-Equation
      Direct_radiance = Direct_radiance * Direct_energy * Direct_visibility;
      Indirect_radiance = Indirect_radiance * Indirect_energy * Indirect_visivility;
      Integrate(camera to endPoint){ transmittance(camera to thePoint) * (Direct_radiance + Indirect_radiance) * (- absorption - outscattering + inscattering) * phase(theta) * ds }
  
  1.2 Volume's Integrated Lighting Energy (Probability)
      Beer's law or Andrew Schneider's powdersugar
  
  1.3 Direct Radiance and Indirect Radiance
      Precompute Atmosphere Scattering
      
  1.4 We need to Consider all important participating media
      ForExample: Atmosphere Cloudsphere Fogsphere Scene ...
      
      1.4.1 Scene:
          Light cannot pass through it
          use approximated-geometry-render-method(Lambert-diffuse, Microfacet-specul, ...)
      
      1.4.2 Atmosphere Cloudsphere Fogsphere:
          Light can pass through it
          use Volume-render-method(Raymarhc-integrate, Eric-Bruneton's Precomputed LookupTable(Must be continuous media, and static))
 
  Reference
    (c) Eric Bruneton Atmosphere Scattering Sky Model 
    URL: "https://github.com/ebruneton/precomputed_atmospheric_scattering"
    License: "BSP-3"
    (c) GPU Pro7 ...
    (c) SIG2015 SIG2017 SIG2019 Advances in Real-Time Render ...

2. Cloud Shape
  2.1 Schneider's Cloud-Model
    Shape = Function(coverage, height_coverage, details)
    
  2.2 Schneider's Source in 2015
    remap(base_noise * height_coverage, 1.0 - coverage, 1.0) * coverage
    
    Important is <Remap idea>, and Remap(value, lower, upper, 0, 1) equal Rescale(value, lower, upper)
    I recommented first use noise and customized-image in 2D-Render
    
    Important is <Fraction idea>, added higher-and-higher frequency noise octive
    Can see <Real-time Volume Render> or <Texturing And Modeling>
  
  2.3 Other-Source in shadertoy by alro
    coverage * hight_coverage = Volume_Container_density
    Cloud_density = rescale(Volume_Container_density，noise)
    Cloud_density = rescale(Cloud_density，highfreq_noise)
    Cloud_density = rescale(Cloud_density，highhighfreq_noise)
    from www.shadertoy.com/view/WscyWB, His great work
    
3. Raymarch
   3.1 Equal-Length-Raymarch
       Length ds = Constant;
       for(int i = 0; i != step; ++i){
          Length s = i * ds;
          ...
       }
       
       Simple method, by increment ds can avoid fault
       
   3.2 Detail-Raymarch
       Length step_length = Constant;
       Length detail_length = small_length;
       bool State = Normal;
       for(int i = 0; i != step; ++i){
          if(State == Detail){
             ...
             Length ds = detail_length;
          } else {
             Length ds = step_length;
             ...
             if(...){
                State = Detail;
             }
          }
       }
       
       More detail results
   
   3.3 Empty-Space-Raymarch
       Length init_step_length = setup(detail_length, n);
       for(int i = 0; i != step; ++i){
           Length ds = pow(0.5,n)*init_step_length;
           ...
       }
      
       Close camera Objects must are very large and continuous
   
   3.4 Structed-Raymarch
       setup(t, dt, wt);
       for(int i = 0; i != step; ++i){
          ds = wt*dt;
          
          ...
          t += dt; // cannot use i * dt
       }
       
       Avoid fault
       
       
I believe you don't need to look at Source after reading this README
