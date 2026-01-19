// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/render/OptixGlobals.h"
#include "barney/umesh/mc/IconField.h"
#include "barney/volume/DDA.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {
  namespace render {

    struct IconField_Programs {
      
      static inline __rtc_device
      void anyHit(rtc::TraceInterface &ti)
      { /* ignore, not used, but has to exist */ }
      
      static inline __rtc_device
      void closestHit(rtc::TraceInterface &ti)
      { /* TODO: getPRD, then set appropriate triangle ID */
        auto &prd = *(IconMultiPassSampler::PRD *)ti.getPRD();
        prd.primID = ti.getPrimitiveIndex();
      }
      
    };


    struct IconField_TraceRays {
#if RTC_DEVICE_CODE
      inline __rtc_device static 
      void run(rtc::TraceInterface &ti);
#endif
    };

#if RTC_DEVICE_CODE
    inline __rtc_device 
    void IconField_TraceRays::run(rtc::TraceInterface &ti)
    {
#ifdef NDEBUG
      enum { dbg = false };
#else
      const bool dbg = ray.dbg();
#endif
      const int rayID
        = ti.getLaunchIndex().x
        + ti.getLaunchDims().x
        * ti.getLaunchIndex().y;
      //if (rayID == 0)
      //  printf("iconfield whole-frame launch ...\n");

      //const IconMultiPassAccel::DD &self
      //    = *(IconMultiPassAccel::DD*)ti.getProgramData();
      auto &lp = OptixGlobals::get(ti);

      auto &self = *(const IconMultiPassAccel::DD *)lp.userData;

      if (rayID >= lp.numRays)
        return;

      Ray &ray = lp.rays[rayID];

      box3f bounds = self.volume.sfCommon.worldBounds;
      //range1f tRange = { ti.getRayTmin(), ti.getRayTmax() };
      range1f tRange = { 0.f, FLT_MAX };
    
      vec3f obj_org = ray.org;//ti.getObjectRayOrigin();
      vec3f obj_dir = ray.dir;//ti.getObjectRayDirection();

      auto objRay = ray;
      objRay.org = obj_org;
      objRay.dir = obj_dir;

      if (!boxTest(objRay,tRange,bounds))
        return;
    
      // ------------------------------------------------------------------
      // compute ray in macro cell grid space 
      // ------------------------------------------------------------------
      vec3f mcGridOrigin  = self.mcGrid.gridOrigin;
      vec3f mcGridSpacing = self.mcGrid.gridSpacing;

      vec3f dda_org = obj_org;
      vec3f dda_dir = obj_dir;

      dda_org = (dda_org - mcGridOrigin) * rcp(mcGridSpacing);
      dda_dir = dda_dir * rcp(mcGridSpacing);

      //Random rng(ray.rngSeed,hash(ti.getRTCInstanceIndex(),
      //                            ti.getGeometryIndex(),0));
      Random rng(ray.rngSeed,hash(0,0,0));

      dda::dda3(dda_org,dda_dir,tRange.upper,
                vec3ui(self.mcGrid.dims),
                [&](const vec3i &cellIdx, float t0, float t1) -> bool
                {
                  const float majorant = self.mcGrid.majorant(cellIdx);
                  
                  if (majorant == 0.f) return true;
                  
                  vec4f   sample = 0.f;
                  range1f tRange = {t0,min(t1,ray.tMax)};
                  if (!Woodcock::sampleRange(sample,
                                             self.volume,
                                             obj_org,
                                             obj_dir,
                                             tRange,
                                             majorant,
                                             rng,
                                             dbg)) 
                    return true;
                  if (dbg) printf("woodcock hit sample %f %f %f:%f\n",
                                  sample.x,
                                  sample.y,
                                  sample.z,
                                  sample.w);
                  
                  vec3f P_obj = obj_org + tRange.upper * obj_dir;
                  vec3f P = P_obj;//ti.transformPointFromObjectToWorldSpace(P_obj);
                  ray.setVolumeHit(P,
                                   tRange.upper,
                                   getPos(sample));
                  //ti.reportIntersection(tRange.upper, 0);
                  return false;
                },
                /*NO debug:*/false
                );
#if 0
      if (rayID >= lp.numRays)
        return;
      
      Ray &ray = lp.rays[rayID];
        
      vec3f dir = ray.dir;
      if (dir.x == 0.f) dir.x = 1e-6f;
      if (dir.y == 0.f) dir.y = 1e-6f;
      if (dir.z == 0.f) dir.z = 1e-6f;

      ti.traceRay(lp.accel,
                  ray.org,
                  dir,
                  0.f,
                  ray.tMax,
                  /* PRD */
                  (void *)&ray);

      box3f b(vec3f(-39587.1,0.164928,6.36584e+06),vec3f(211866,186394,6.37164e+06));
      float t0,t1;
      boxTest(t0,t1,b,ray.org,ray.dir);
      //if (ray.hadHit()) {
      //  printf("%f\n",ray.tMax);
      //}
      if (t0<t1) {
        //printf("t0:%f\n",t0);
        ray.tMax = t0;
      }
#endif
    }

#endif

  }

  // using IconField = MCVolumeAccel<UMeshCuBQLSampler>;
  // using IconField_Iso = MCIsoSurfaceAccel<UMeshCuBQLSampler>;

  RTC_EXPORT_TRIANGLES_GEOM(IconField,IconMultiPassSampler::DD,
                            render::IconField_Programs,false,true);
  RTC_EXPORT_TRACE2D(traceRays_IconField,render::IconField_TraceRays);
}



