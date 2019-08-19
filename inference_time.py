			
      prev_time = time.time()

      ...
      
			curr_time = time.time()
      exec_time = curr_time - prev_time

			info = "time:" + str(round(1000*exec_time, 2)) + " ms, FPS: " + str(round((1000/(1000*exec_time)),1))
			
      print(info)
