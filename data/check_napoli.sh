#!/bin/bash
#  napoli_job_caller.sh
#  Author: Sasha Sax
#  
#  Desc: This will process all the models in the dataset using the provided command.
#   It uses the Napoli cluster. After running this command it will prompt for username
#   and password. Then it will ssh into all the machines in the cluster and dispatch
#   jobs, running them in a screen on the machine. 
#
#  Usage:
#    bash napoli_job_caller.sh

# Set job parameters

JOB_NAME="generate_data"


NAPOLI_CPU_MIN=101
NAPOLI_CPU_MAX=110
# max_task_idx=15 #$((NAPOLI_CPU_MAX-NAPOLI_CPU_MIN))


# Get credentials
# Read User
echo -n User: 
read user
# Read Password
echo -n Password: 
read -s password
echo



echo Checking $JOB_NAME tasks on Napoli $NAPOLI_CPU_MIN
ssh $user@napoli${NAPOLI_CPU_MIN} <<-EOI
    cd /cvgl/u/taskonomy/preprocessing/jobs/$JOB_NAME
    export max_task_idx=\$(ls *_unprocessed_ids.txt | wc -l)
    export max_task_idx=\$((max_task_idx-1))
    export grand_total_tasks=0
    export grand_total_completed=0
    export grand_total_errors=0
    for i in \$(seq 0 \$max_task_idx); do
        export total_tasks=\$(wc -l \${i}_unprocessed_ids.txt | cut -f1 -d' ')
        export failed_tasks=\$(wc -l \${i}_failed.txt 2> /dev/null | cut -f1 -d' ')
        export completed_tasks=\$(wc -l \${i}_processed_ids.txt 2> /dev/null | cut -f1 -d' ')
        if [ -z \$failed_tasks ]; then export failed_tasks=0; fi
        if [ -z \$completed_tasks ]; then export completed_tasks=0; fi
        echo -e "\\tTask \$i: \$completed_tasks/\$total_tasks (\$failed_tasks failed)"

        export grand_total_tasks=\$((grand_total_tasks+total_tasks))
        export grand_total_completed=\$((grand_total_completed+completed_tasks))
        export grand_total_errors=\$((grand_total_errors+failed_tasks))
    done
    echo Grand total: \$grand_total_completed/\$grand_total_tasks \(\$grand_total_errors failed\) 
EOI
