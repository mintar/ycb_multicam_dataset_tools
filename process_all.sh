#!/bin/bash
find -name _camera_settings.json -print0 | xargs -0 -L 1 -I '{}' bash -c 'echo -e \\n\\n\\n\\n\\n\\n\\n=================== "$(dirname {})" && cd "$(dirname {})" && process_single_folder.sh'
