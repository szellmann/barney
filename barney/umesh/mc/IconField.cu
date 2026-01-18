// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barney/umesh/mc/IconField.h"
#include "rtcore/ComputeInterface.h"
#include <cub/cub.cuh>

namespace BARNEY_NS {
  using render::OptixGlobals;

  /* helpers */
  inline __host__ __device__ vec3f toSpherical(const vec3f cartesian)
  {
    float r = length(cartesian);
    float lat = asinf(cartesian.z/r);
    float lon = atan2f(cartesian.y, cartesian.x);
    return {r,lat,lon};
  }

  inline __host__ __device__ vec3f toCartesian(const vec3f spherical)
  {
    const float r = spherical.x;
    const float lat = spherical.y;
    const float lon = spherical.z;

    float x = r * cosf(lat) * cosf(lon);
    float y = r * cosf(lat) * sinf(lon);
    float z = r * sinf(lat);
    return {x,y,z};
  }

  __host__ __device__
  inline unsigned morton_encode2D(unsigned x, unsigned y)
  {
      auto separate_bits = [](unsigned n)
      {
          n &= 0x0000FFFF;
          n = (n ^ (n <<  8)) & 0x00FF00FF;
          n = (n ^ (n <<  4)) & 0x0F0F0F0F;
          n = (n ^ (n <<  2)) & 0x33333333;
          n = (n ^ (n <<  1)) & 0x55555555;
          return n;
      };
  
      return separate_bits(x) | (separate_bits(y) << 1);
  }

  __host__ __device__
  inline unsigned morton_encode3D(unsigned x, unsigned y, unsigned z)
  {
    auto separate_bits = [](unsigned n) {
      n &= 0x000003FF;
      n = (n ^ (n << 16)) & 0xFF0000FF;
      n = (n ^ (n <<  8)) & 0x0300F00F;
      n = (n ^ (n <<  4)) & 0x030C30C3;
      n = (n ^ (n <<  2)) & 0x09249249;
      return n;
    };  
  
    return separate_bits(x) | (separate_bits(y) << 1) | (separate_bits(z) << 2); 
  }

  __host__ __device__
  inline unsigned long long morton_encode3D(unsigned long long x, unsigned long long y, unsigned long long z)
  {
    auto separate_bits = [](unsigned long long n) {
      n &= 0b1111111111111111111111ull;
      n = (n ^ (n << 32)) & 0b1111111111111111000000000000000000000000000000001111111111111111ull;
      n = (n ^ (n << 16)) & 0b0000000011111111000000000000000011111111000000000000000011111111ull;
      n = (n ^ (n <<  8)) & 0b1111000000001111000000001111000000001111000000001111000000001111ull;
      n = (n ^ (n <<  4)) & 0b0011000011000011000011000011000011000011000011000011000011000011ull;
      n = (n ^ (n <<  2)) & 0b1001001001001001001001001001001001001001001001001001001001001001ull;
      return n;
    };  
  
    return separate_bits(x) | (separate_bits(y) << 1) | (separate_bits(z) << 2); 
  }


  /* single layer of ICON data: */
  struct ICONLayer
  {
    // Morton code of base triangle centroid:
    uint64_t mortonID;
    // lon/lat coordinates of base
    vec3f lon, lat;
    // height of *this layer*
    float height;
    // value of *this layer*
    float value;
  };

  __host__ __device__
  inline bool compLonLat(const ICONLayer &l1, const ICONLayer &l2) {
    auto fix_P1 = [](float &f) {
      float ff = f;
      if (ff > 1e-5f) ff-=M_PI;
      if (ff > 1e-5f) ff-=M_PI;
      ff = roundf(ff);
      if (ff == 0.f) f = ff;
    };

    auto fix_P3 = [=](vec3f &v) {
      fix_P1(v.x);
      fix_P1(v.y);
      fix_P1(v.z);
    };

    vec3f diff_lon(
        fabsf(l1.lon.x-l2.lon.x),
        fabsf(l1.lon.y-l2.lon.y),
        fabsf(l1.lon.z-l2.lon.z));
    fix_P3(diff_lon);
    if (length(diff_lon) > 1e-3f) return false;

    vec3f diff_lat(
        fabsf(l1.lat.x-l2.lat.x),
        fabsf(l1.lat.y-l2.lat.y),
        fabsf(l1.lat.z-l2.lat.z));
    fix_P3(diff_lat);
    if (length(diff_lat) > 1e-3f) return false;

    return true;
  }

  struct CompareMorton
  {
    __host__ __device__ bool operator()(const ICONLayer &a, const ICONLayer &b)
    { return a.mortonID<b.mortonID; }
  };

  __global__ void computeLayers(ICONLayer *layers, const UMeshField::DD &field)
  {
    int cellID = threadIdx.x+blockIdx.x*blockDim.x;
    if (cellID >= field.numCells)
      return;
 
    uint8_t cellType = field.cellTypes[cellID];
    assert(cellType==_ANARI_PRISM||cellType==_VTK_PRISM);

    const int *I = field.indices + field.cellOffsets[cellID];
    const vec3f v0 = field.vertices[I[0]];
    const vec3f v1 = field.vertices[I[1]];
    const vec3f v2 = field.vertices[I[2]];
    const vec3f v3 = field.vertices[I[3]];
    const vec3f v4 = field.vertices[I[4]];
    const vec3f v5 = field.vertices[I[5]];

    // cartesian centroids:
    const vec3f c0 = (v0+v1+v2)/3.f;
    const vec3f c1 = (v3+v4+v5)/3.f;

    // spherical coordinates
    const vec3f sv0 = toSpherical(v0);
    const vec3f sv1 = toSpherical(v1);
    const vec3f sv2 = toSpherical(v2);

    //if (cellID<=10) {
    //  printf("%i %f,%f,%f %f,%f,%f %f,%f,%f\n",
    //      cellID,
    //      sv0.x,sv0.y,sv0.z,
    //      sv1.x,sv1.y,sv1.z,
    //      sv2.x,sv2.y,sv2.z);
    //}
    //const vec3f sv3 = toSpherical(v3);
    //const vec3f sv4 = toSpherical(v4);
    //const vec3f sv5 = toSpherical(v5);

    // spherical centroids:
    const vec3f sc0 = (sv0+sv1+sv2)/3.f;
    //const vec3f sc1 = (sv3+sv4+sv5)/3.f;

    // values:
    const float valueBot = field.scalars[I[0]];
    const float valueTop = field.scalars[I[3]];
    //const float valueBot = field.scalars[cellID];
    //const float valueTop = field.scalars[cellID];

    // Morton code for base triangle:
    const box3f bounds = field.worldBounds;
    const vec3f qc0 = (c0-bounds.lower)/bounds.size();
    const vec2f qsc0(sc0.y/(2*M_PI),sc0.z/(2*M_PI));
    

    // construct layer:
    ICONLayer &layer = layers[cellID];
    layer.lon = vec3f(sv0.z,sv1.z,sv2.z);
    layer.lat = vec3f(sv0.y,sv1.y,sv2.y);
    layer.height = sc0.x;//length(c0);
    layer.value = (valueBot+valueTop)*0.5f;
    // quantize, but avoid too much precision, otherwise
    // the morton codes will be falsely different......
    //layer.mortonID = morton_encode3D((unsigned long long)(qc0.x*(2<<20)),
    //                                 (unsigned long long)(qc0.y*(2<<20)),
    //                                 (unsigned long long)(qc0.z*(2<<20)));
    layer.mortonID = morton_encode3D(0ull,
                                     (unsigned long long)(qsc0.x*(2<<20)),
                                     (unsigned long long)(qsc0.y*(2<<20)));

    // if (cellID<15) {
    //   printf("%f,%f,%f\n",qc0.x,qc0.y,qc0.z);
    //   printf("Cell %i -- centroid (bot): %f,%f,%f, morton(centroid): %u, lon: %f,%f,%f, lat: %f,%f,%f, height: %f, value: %f\n",
    //       cellID,
    //       sc0.x,sc0.y,sc0.z,
    //       layer.mortonID,
    //       layer.lon.x,layer.lon.y,layer.lon.z,
    //       layer.lat.x,layer.lat.y,layer.lat.z,
    //       layer.height,
    //       layer.value);
    // }
  }

  __global__ void guessNumLayers(int *minLayers, int *maxLayers,
                                 const ICONLayer *layers, int numCells)
  {
    int cellID = threadIdx.x+blockIdx.x*blockDim.x;
    if (cellID >= numCells)
      return;

    const ICONLayer &layer = layers[cellID];
    uint64_t mortonID = layer.mortonID;

    int left = cellID;
    while (left > 0) {
      if (!compLonLat(layers[left-1],layer)) break;
      left--;
      if (cellID-left > 120) break;
    }

    int right = left+1;
    while (right < numCells) {
      if (!compLonLat(layers[right],layer)) break;
      right++;
      if (right-cellID > 120) break;
    }

    int numLayers = right-left;

    atomicMin(minLayers,numLayers);
    atomicMax(maxLayers,numLayers);
  }

  __global__ void mergeLayers(ICONCell *cells, ICONLayer *layers, int numICONCells, int layersPerCell)
  {
    int cellID = threadIdx.x+blockIdx.x*blockDim.x;
    if (cellID >= numICONCells)
      return;

    int layerID = cellID * layersPerCell;

    for (int l=0; l<layersPerCell; ++l) {
      cells[cellID].height[l] = FLT_MAX;
    }

    // insert height/values pairs sorted:
    for (int l=0; l<layersPerCell; ++l) {
      const ICONLayer &layer = layers[layerID*layersPerCell+l];
      int idx=0;
      while (cells[cellID].height[idx] < layer.height) {
        idx++;
      }
      assert(idx < ICONCell::MaxLayers-1);
      for (int l=layersPerCell; l>idx; --l) {
        cells[cellID].height[l] = cells[cellID].height[l-1];
        cells[cellID].value[l] = cells[cellID].value[l-1];
      }
      cells[cellID].height[idx] = layer.height;
      cells[cellID].value[idx] = layer.value;
    }

    // for (int l=0; l<layersPerCell; ++l) {
    //   if(cellID==3) {
    //     printf("cellID: %i,height: %f, value: %f\n",
    //         cellID,cells[cellID].height[l],cells[cellID].value[l]);
    //   }
    // }
  }

  RTC_IMPORT_TRIANGLES_GEOM(/*file*/IconField,/*name*/IconField,
                            /*geomtype device data */
                            IconMultiPassSampler::DD,false,true);
  RTC_IMPORT_TRACE2D
  (/*IconField.cu*/IconField,
   /*ray gen name */traceRays_IconField,
   /*launch params data type*/sizeof(BARNEY_NS::render::OptixGlobals)
   );
  
  
  void IconMultiPassAccel::build(bool full_rebuild)
  {
    if (!majorantsGrid) {
      auto mcGrid = volume->sf->getMCs();
      majorantsGrid = std::make_shared<MajorantsGrid>(mcGrid);
    }
    majorantsGrid->computeMajorants(&volume->xf);
    sfSampler->build();

    auto thisPass = std::make_shared<IconMultiPassLaunch>
      (sampler);
    volume->generatedPasses = { thisPass };
    
#if 0
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);

      auto creatorFunction = createGeomType_IconField;
      // build our own internal per-device data: one geom, and one
      // group that contains it.
      PLD *pld = getPLD(device);
      if (!pld->geom) {
        rtc::GeomType *gt
          = device->geomTypes.get(creatorFunction);
        // build a single-prim geometry, that single prim is our
        // entire MC/DDA grid
        pld->geom = gt->createGeom();
        pld->geom->setPrimCount(1);
      }
      rtc::Geom *geom = pld->geom;
      DD dd = getDD(device);
      geom->setDD(&dd);
      
      if (!pld->group) {
        // now put that into a instantiable group, and build it.
        pld->group = device->rtc->createTrianglesGroup({geom});
        // pld->group = device->rtc->createUserGeomsGroup({geom});
      }
      pld->group->buildAccel();
      
      // now let the actual volume we're building know about the
      // group we just created
      Volume::PLD *volumePLD = volume->getPLD(device);
      if (volumePLD->generatedGroups.empty()) 
        volumePLD->generatedGroups = { pld->group };
    }
#endif
  }

  void IconMultiPassLaunch::launch(Device *device,
                                   const render::World::DD &world,
                                   const affine3f &instanceXfm,
                                   render::Ray *rays,
                                   int numRays)
  {
    // PING;
    int bs = 128;
    int nb = divRoundUp(numRays,bs);
    auto rayGen = sampler->getPLD(device)->rayGen;

    OptixGlobals dd;
    dd.world = world;
    dd.rays = rays;
    dd.numRays = numRays;
    dd.accel = sampler->getPLD(device)->baseTrisTLAS->getDD();
    dd.userData = (void *)sampler->getPLD(device)->cells;
    rayGen->launch(/* bs,nb intentionally inverted:
                      always have 1024 in width: */
                   vec2i(bs,nb),
                   &dd);cudaDeviceSynchronize();
  }

  IconMultiPassAccel::IconMultiPassAccel(Volume *volume,
                                         IconMultiPassSampler::SP sampler)
    : MCVolumeAccel<IconMultiPassSampler>(volume,nullptr,sampler),
      sampler(sampler)
  {
    PING;
  }





  
  IconMultiPassSampler::IconMultiPassSampler(UMeshField *field)
    : perLogical(field->devices->size()),
      field(field)
  {
    PING;
  }

  IconMultiPassSampler::PLD *IconMultiPassSampler::getPLD(Device *device)
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }

  IconMultiPassSampler::DD IconMultiPassSampler::getDD(Device *device)
  {
    PING;
    return { getPLD(device)->baseTrisTLAS->getDD() };
  }
    
  void IconMultiPassSampler::build()
  {
    PING;

    // purely for testing:
    box3f bounds = field->worldBounds;
    PRINT(bounds);

    vec3f v0 = bounds.lower;
    vec3f v1 = bounds.upper;
    vec3f v2 = {
      bounds.lower.x,
      bounds.lower.y,
      bounds.upper.z
    };

    auto creatorFunction = createGeomType_IconField;
    for (auto device : *field->devices) { 
      auto rtc = device->rtc;
      PLD *pld = getPLD(device);

      if (!pld->cells) {
        ICONLayer *d_layers{nullptr};
        BARNEY_CUDA_CALL(Malloc(&d_layers, sizeof(ICONLayer)*field->numCells));
        computeLayers<<<divRoundUp(field->numCells,1024),1024>>>(
              d_layers, field->getDD(device));

        // Sort layers refs by morton codes
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceMergeSort::StableSortKeys(
            d_temp_storage,
            temp_storage_bytes,
            d_layers,
            field->numCells,
            CompareMorton()
            );
        BARNEY_CUDA_CALL(Malloc(&d_temp_storage, temp_storage_bytes));
        cub::DeviceMergeSort::StableSortKeys(
            d_temp_storage,
            temp_storage_bytes,
            d_layers,
            field->numCells,
            CompareMorton()
            );
        BARNEY_CUDA_CALL(Free(d_temp_storage));

        int *d_minLayers, *d_maxLayers;
        BARNEY_CUDA_CALL(Malloc(&d_minLayers, sizeof(int)));
        BARNEY_CUDA_CALL(Malloc(&d_maxLayers, sizeof(int)));

        int minLayers{INT_MAX}, maxLayers{0};
        BARNEY_CUDA_CALL(Memcpy(d_minLayers, &minLayers, sizeof(minLayers), cudaMemcpyHostToDevice));
        BARNEY_CUDA_CALL(Memcpy(d_maxLayers, &maxLayers, sizeof(maxLayers), cudaMemcpyHostToDevice));

        guessNumLayers<<<divRoundUp(field->numCells,1024),1024>>>(
              d_minLayers, d_maxLayers, d_layers, field->numCells);

        BARNEY_CUDA_CALL(Memcpy(&minLayers, d_minLayers, sizeof(minLayers), cudaMemcpyDeviceToHost));
        BARNEY_CUDA_CALL(Memcpy(&maxLayers, d_maxLayers, sizeof(maxLayers), cudaMemcpyDeviceToHost));
        std::cout << "Seems we have [min:max] layers: [" << minLayers << ':' << maxLayers << "]\n";

        if (minLayers != maxLayers) {
          // TODO....:
          std::cerr << "That doesn't match......\n";
          exit(0);
        }

        int numICONCells = field->numCells/minLayers;

        // Merge layers
        BARNEY_CUDA_CALL(Malloc(&pld->cells, numICONCells*sizeof(ICONCell)));
        mergeLayers<<<divRoundUp(numICONCells,1024),1024>>>(
              pld->cells, d_layers, numICONCells, minLayers);

        BARNEY_CUDA_CALL(Free(d_layers));
        pld->numCells = numICONCells;
      }
    
      if (!pld->baseTrisTLAS) {
        // TODO: convdert on the device (does rtc::Buffer support that?):
        std::vector<vec3f> vtx;
        std::vector<vec3i> idx;
        std::vector<ICONCell> hCells(pld->numCells);
        BARNEY_CUDA_CALL(Memcpy(hCells.data(), pld->cells,
                                pld->numCells*sizeof(ICONCell),
                                cudaMemcpyDeviceToHost));
        for (size_t i=0; i<hCells.size(); ++i) {
          const ICONCell &cell = hCells[i];
          vec3f v1 = toCartesian({cell.height[0],cell.lat.x,cell.lon.x});
          vec3f v2 = toCartesian({cell.height[0],cell.lat.y,cell.lon.y});
          vec3f v3 = toCartesian({cell.height[0],cell.lat.z,cell.lon.z});
          vtx.push_back(v1);
          vtx.push_back(v2);
          vtx.push_back(v3);
          idx.push_back({int(i)*3,int(i)*3+1,int(i)*3+2});
        }

        // create a rtc group (ie tlas) for the given object that we
        // can trace rays against, over a single triangle mesh
        rtc::Buffer *vertices = rtc->createBuffer(vtx.size()*sizeof(vec3f),vtx.data());
        rtc::Buffer *indices = rtc->createBuffer(idx.size()*sizeof(vec3i),idx.data());
        rtc::GeomType *gt
          = device->geomTypes.get(creatorFunction);
        rtc::Geom *geom
          = gt->createGeom();
        geom->setPrimCount(idx.size());
        geom->setVertices(vertices, vtx.size());
        geom->setIndices(indices, idx.size());
        rtc::Group *blas = rtc->createTrianglesGroup({geom});
        blas->buildAccel();
        rtc::Group *tlas = rtc->createInstanceGroup({blas},{},{});
        tlas->buildAccel();
        
        pld->baseTrisTLAS = tlas;

        
        pld->rayGen = createTrace_traceRays_IconField(device->rtc);
      }
    }
  }


}
