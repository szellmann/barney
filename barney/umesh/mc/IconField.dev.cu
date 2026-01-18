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
      const int rayID
        = ti.getLaunchIndex().x
        + ti.getLaunchDims().x
        * ti.getLaunchIndex().y;
      //if (rayID == 0)
      //  printf("iconfield whole-frame launch ...\n");

      auto &lp = OptixGlobals::get(ti);

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
      if (ray.tMax < 100000000.f) {
        printf("%f\n",ray.tMax);
      }
      if (t0<t1) {
        //printf("t0:%f\n",t0);
        ray.tMax = t0;
      }
    }
#endif

  }

  // using IconField = MCVolumeAccel<UMeshCuBQLSampler>;
  // using IconField_Iso = MCIsoSurfaceAccel<UMeshCuBQLSampler>;

  RTC_EXPORT_TRIANGLES_GEOM(IconField,IconMultiPassSampler::DD,
                            render::IconField_Programs,false,true);
  RTC_EXPORT_TRACE2D(traceRays_IconField,render::IconField_TraceRays);
}



