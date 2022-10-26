
## summary ##
-----------

### log format ###
```
[time][level][thread_id][code location] log_content
|   auto-generated(support config)    | | content |
```

### log example ###
```
[2022-10-21 09:47:36.835][Debug][7fff43722700][../nodes/vp_node.cpp:179] [screen_des_a] after meta flow, in_queue.size()==>12
```

### tips ###
better to add important field using `[]` in log content `Manually`, such as `module`, `type`. below code add name of host node(module) and task(type) in log content.

```
VP_INFO(vp_utils::string_format("[%s] [record] save dir not exists, now creating save dir: `%s`", host_node_name, save_dir));
```

### log api ###

#### log config ####
```c++
// log level
VP_SET_LOG_LEVEL(_log_level);
// log file dir
VP_SET_LOG_DIR(_log_dir);

// log to console or not
VP_SET_LOG_TO_CONSOLE(_log_to_console);
// log to file or not
VP_SET_LOG_TO_FILE(_log_to_file);
// TO-DO
VP_SET_LOG_TO_KAFKA(_log_to_kafka);

// include log level or not
VP_SET_LOG_INCLUDE_LEVEL(_include_level);
// include code location or not (where the log occurs)
VP_SET_LOG_INCLUDE_CODE_LOCATION(_include_code_location);
// include thread id or not (std::this_thread::get_id())
VP_SET_LOG_INCLUDE_THREAD_ID(_include_thread_id);

// warn if log cache in memory exceed this value
VP_SET_LOG_CACHE_WARN_THRES(_log_cache_warn_threshold);
```


#### write log ####
4 types of log
```c++
// error
VP_ERROR(message);
// warn
VP_WARN(message);
// info
VP_INFO(message);
// debug
VP_DEBUG(message);
```

```c++
// important! call at the begining of main()
VP_LOGGER_INIT();
```
