# dreamworld1
Dream World 1, Volume Cloud

I believe you don't need to look at Source after reading this README



Teams:
Probability: { 0 <= x <= 1 }

1. Lighting 
  1.1 Volume's Integrated Lighting Energy (Probability)
      Beer's law or Andrew Schneider's powdersugar
  
  1.2 Direct Lighting and Indirect Lighting
      Precompute Atmosphere Scattering
      
  1.3 We need to Consider all important participating media
      ForExample: Atmosphere Cloudsphere Fogsphere Scene ...
      
      1.3.1 Scene:
          Light cannot pass through it
          use approximated-geometry-render-method(Lambert-diffuse, Microfacet-specul, ...)
      
      1.3.2 Atmosphere Cloudsphere Fogsphere:
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
    
  2.1 Schneider's Source in 2015
    remap(base_noise * height_coverage, 1.0 - coverage, 1.0) * coverage
    
    Important is <Remap idea>, and Remap(value, lower, upper, 0, 1) equal Rescale(value, lower, upper)
    I recommented first use noise and customized-image in 2D-Render
    
    Important is <Fraction idea>, added higher-and-higher frequency noise octive
    Can see <Real-time Volume Render> or <Texturing And Modeling>
  
  2.2 Other-Source in shadertoy
    coverage * hight_coverage = Volume_Container_density
    Cloud_density = rescale(Volume_Container_density，noise)
    Cloud_density = rescale(Cloud_density，highfreq_noise)
    Cloud_density = rescale(Cloud_density，highhighfreq_noise)
    from www.shadertoy.com/view/WscyWB, His great work
