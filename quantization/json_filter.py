
failed_tasks = {
  "code": 0,
  "msg": "",
  "detail": [
    {
      "pkg_task_id": "999180333210000000161202285_null_210000000161202285",
      "status": -1,
      "status_landmark": 2,
      "status_land_line": 2,
      "status_scene": 2,
      "status_camera": 2,
      "status_traffic_light": -1,
      "status_img": -1
    },
    {
      "pkg_task_id": "999180100210000000177231923_null_210000000177231923",
      "status": -1,
      "status_landmark": 2,
      "status_land_line": 2,
      "status_scene": 2,
      "status_camera": 2,
      "status_traffic_light": -1,
      "status_img": -1
    },
    {
      "pkg_task_id": "999180203210000000161217375_null_210000000161217375",
      "status": -1,
      "status_landmark": 2,
      "status_land_line": 2,
      "status_scene": 2,
      "status_camera": 2,
      "status_traffic_light": -1,
      "status_img": -1
    },
    {
      "pkg_task_id": "999180148210000000161242266_null_210000000161242266",
      "status": -1,
      "status_landmark": 2,
      "status_land_line": 2,
      "status_scene": 2,
      "status_camera": 2,
      "status_traffic_light": -1,
      "status_img": -1
    },
    {
      "pkg_task_id": "999204618210000000161212957_null_210000000161212957",
      "status": -1,
      "status_landmark": 2,
      "status_land_line": 2,
      "status_scene": 2,
      "status_camera": 2,
      "status_traffic_light": -1,
      "status_img": -1
    },
    {
      "pkg_task_id": "999195621210000000177226766_null_210000000177226766",
      "status": -1,
      "status_landmark": 2,
      "status_land_line": 2,
      "status_scene": 2,
      "status_camera": 2,
      "status_traffic_light": -1,
      "status_img": -1
    },
    {
      "pkg_task_id": "999202427210000000161237174_null_210000000161237174",
      "status": -1,
      "status_landmark": 2,
      "status_land_line": 2,
      "status_scene": 2,
      "status_camera": 2,
      "status_traffic_light": -1,
      "status_img": -1
    },
    {
      "pkg_task_id": "999202404210000000177220569_null_210000000177220569",
      "status": -1,
      "status_landmark": 2,
      "status_land_line": 2,
      "status_scene": 2,
      "status_camera": 2,
      "status_traffic_light": -1,
      "status_img": -1
    },
    {
      "pkg_task_id": "999192126210000000177226943_null_210000000177226943",
      "status": -1,
      "status_landmark": 2,
      "status_land_line": 2,
      "status_scene": 2,
      "status_camera": 2,
      "status_traffic_light": -1,
      "status_img": -1
    },
    {
      "pkg_task_id": "999194214210000000177357511_null_210000000177357511",
      "status": -1,
      "status_landmark": 2,
      "status_land_line": 2,
      "status_scene": 2,
      "status_camera": 2,
      "status_traffic_light": -1,
      "status_img": -1
    },
    {
      "pkg_task_id": "999194403210000000177230387_null_210000000177230387",
      "status": -1,
      "status_landmark": 2,
      "status_land_line": 2,
      "status_scene": 2,
      "status_camera": 2,
      "status_traffic_light": -1,
      "status_img": -1
    }
  ],
  "detail_explain": "pkg_task_id,status,status_landmark,status_land_line,status_scene,status_camera,status_traffic_light,status_img"
}

for info in failed_tasks["detail"]:
    print('"{}"'.format(info["pkg_task_id"]), end = ",")

print("\n"*4)