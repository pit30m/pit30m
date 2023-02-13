"""Holding area for outdated code before fully deleting it."""

    def index_lidar(
        self,
        log_id: str,
        lidar_name: str = "hdl64e_12_middle_front_roof",
        sweep_time_convention: str = "end",
        out_index_fpath: Optional[str] = None,
        check: bool = True,
    ):
        """Same as 'index_all_cameras', except for the LiDAR sweeps."""
        in_fs = fsspec.filesystem(urlparse(self._root).scheme)
        out_fs = fsspec.filesystem(urlparse(out_index_fpath).scheme)
        log_root = os.path.join(self._root, log_id.lstrip("/"))

        if not os.path.isdir(log_root):
            raise RuntimeError(f"Log {log_id} directory does not exist at all. Indexing failed.")

        log_reader = LogReader(log_root_uri=log_root)
        lidar_dir = self.get_lidar_dir(log_root, lidar_name)
        etl_canary = os.path.join(lidar_dir, ETL_CANARY_FNAME)

        if out_index_fpath is None:
            out_index_fpath = os.path.join(lidar_dir, "index")

        wgs84_index_fpath = os.path.join(out_index_fpath, "raw_wgs84.csv")
        utm_index_fpath = os.path.join(out_index_fpath, "utm.csv")
        unindexed_fpath = os.path.join(out_index_fpath, "unindexed.csv")
        dumped_ts_fpath = os.path.join(lidar_dir, "timestamps.npz.lz4")
        report_fpath = os.path.join(out_index_fpath, "report.csv")

        # Non-sorted list of outputs used in reporting if check is True.
        stats = []

        if not os.path.isfile(etl_canary):
            raise RuntimeError(
                f"Log {log_id} was not dumped yet as its 'ETL finished' canary file was not found; " "can't index it."
            )

        def _get_lidar_time(lidar_uri):
            try:
                with in_fs.open(lidar_uri, "rb") as compressed_f:
                    with lz4.frame.open(compressed_f, "rb") as f:
                        lidar_data = np.load(f)
                        point_times = lidar_data["seconds"]

                        if lidar_data["points"].ndim != 2 or lidar_data["points"].shape[-1] != 3:
                            return (
                                "Error",
                                lidar_uri,
                                "unexpected-points-shape",
                                "{}".format(lidar_data["points"].shape),
                            )
                        if lidar_data["points"].dtype != np.float32:
                            return "Error", lidar_uri, "unexpected-points-dtype", str(lidar_data["points"].dtype)
                        if lidar_data["points_H_sensor"].ndim != 2 or lidar_data["points_H_sensor"].shape[-1] != 3:
                            return (
                                "Error",
                                lidar_uri,
                                "unexpected-points_H_sensor-shape",
                                str(lidar_data["points_H_sensor"].shape),
                            )
                        if lidar_data["points_H_sensor"].dtype != np.float32:
                            return (
                                "Error",
                                lidar_uri,
                                "unexpected-points_H_sensor-dtype",
                                str(lidar_data["points_H_sensor"].dtype),
                            )
                        if len(lidar_data["intensity"]) != len(lidar_data["points"]):
                            return (
                                "Error",
                                lidar_uri,
                                "unexpected-intensity-shape",
                                "{} vs. {} points".format(lidar_data["intensity"].shape, lidar_data["points"].shape),
                            )
                        if len(lidar_data["seconds"]) != len(lidar_data["points"]):
                            return (
                                "Error",
                                lidar_uri,
                                "unexpected-point-time-shape",
                                "{} vs. {} points".format(lidar_data["seconds"].shape, lidar_data["points"].shape),
                            )

                        return (
                            "OK",
                            lidar_uri,
                            point_times.min(),
                            point_times.max(),
                            point_times.mean(),
                            np.median(point_times),
                            lidar_data["points"].shape,
                        )
            except EOFError as err:
                return "Error", lidar_uri, "EOFError", str(err)
            except Exception as err:
                return "Error", lidar_uri, "unknown-error", str(err)

        # TODO(andrei): Directly using the timestamps file seems difficult to leverage as the number of timestamps seems
        # to differ from the number of dumped sweeps, so aligning the two would be challenging. Perhaps I could just use
        # this data to check some of my assumptions later.
        #
        # with in_fs.open(dumped_ts_fpath, "rb") as compressed_f:
        #     with lz4.frame.open(compressed_f, "rb") as f:
        #         timestamps = np.load(f)["data"]

        sample_uris = in_fs.glob(os.path.join(lidar_dir, "*", "*.npz.lz4"))
        print(f"Will analyze and index {len(sample_uris)} samples")
        pool = Parallel(n_jobs=-1, verbose=10)
        time_stats = pool(delayed(_get_lidar_time)(lidar_uri) for lidar_uri in sample_uris)
        raw_wgs84 = log_reader.raw_wgs84_poses_dense

        sweep_times_raw = []
        valid_sample_uris = []
        for sample_uri, result in zip(sample_uris, time_stats):
            status = result[0]
            if status != "OK":
                err_msg = f"error_{str(result[2:])}"
                print(err_msg, sample_uri)
                stats.append(tuple([sample_uri, -1] + list(result[2:])))
                continue

            (min_s, max_s, mean_s, med_s, shape) = result[2:]
            sweep_delta_s = max_s - min_s
            if abs(sweep_delta_s - 0.1) > 0.01:
                print(f"{sample_uri}: sweep_delta_s = {sweep_delta_s:.4f}s | pcd.{shape = }")

            valid_sample_uris.append(sample_uri)
            if sweep_time_convention == "end":
                sweep_times_raw.append(max_s)
            elif sweep_time_convention == "start":
                sweep_times_raw.append(min_s)
            elif sweep_time_convention == "mean":
                sweep_times_raw.append(mean_s)
            elif sweep_time_convention == "median":
                sweep_times_raw.append(med_s)
            else:
                raise ValueError("Unknown sweep time convention: " + sweep_time_convention)

            stats.append((sample_uri, sweep_times_raw[-1], "OK", "n/A"))

        sweep_times = np.array(sweep_times_raw)
        # del sample_uri

        # TODO(andrei): Index by MRP!
        # poses = log_reader.raw_poses
        # pose_data = []
        # for pose in poses:
        #     pose_data.append((pose["capture_time"],
        #                     pose["poses_and_differentials_valid"],
        #                     pose["continuous"]["x"],
        #                     pose["continuous"]["y"],
        #                     pose["continuous"]["z"]))
        # pose_index = np.array(sorted(pose_data, key=lambda x: x[0]))
        # pose_times = np.array(pose_index[:, 0])

        wgs84_times = raw_wgs84[:, 0]
        wgs84_corr_idx = associate(sweep_times, wgs84_times)
        wgs84_delta = abs(raw_wgs84[wgs84_corr_idx, 0] - sweep_times)
        # Recall WGS84 messages are at 10Hz so we have to be a bit more lax than when checking pose assoc
        bad_offs = wgs84_delta > 0.10
        print(bad_offs.sum(), "bad offsets")
        # if bad_offs.sum() > 0:
        #     print(np.where(bad_offs))
        #     print()

        lidars_with_wgs84 = []
        assert len(sweep_times) == len(bad_offs) == len(wgs84_corr_idx)
        assert len(valid_sample_uris) == len(sweep_times)
        for sweep_uri, sweep_time, wgs84_delta_sample, wgs84_idx in tqdm(
            zip(valid_sample_uris, sweep_times, wgs84_delta, wgs84_corr_idx),
            mininterval=TQDM_MIN_INTERVAL_S,
        ):
            bad_off = wgs84_delta_sample > 0.10
            if bad_off:
                stats.append((sweep_uri, sweep_time, "bad-raw-WGS84-offset", f"{wgs84_delta_sample:.4f}s"))
                # TODO Should we flag these in the index?
                continue

            lidar_fpath = "/".join(sweep_uri.split("/")[-2:])

            # img row would include capture seconds, path, then other elements
            # imgs_with_wgs84.append((wgs84_data[wgs84_idx], img_row))
            lidars_with_wgs84.append((raw_wgs84[wgs84_idx, :], (sweep_time, lidar_fpath)))

        # NOTE(andrei): For some rare sweeps (most first sweeps in a log) this will have gaps.
        # NOTE(andrei): The LiDAR is motion-compensated. TBD which timestamp is the canonical one.
        if not out_fs.exists(out_index_fpath):
            out_fs.mkdir(out_index_fpath)
        with out_fs.open(wgs84_index_fpath, "w", newline="") as csvfile:
            spamwriter = csv.writer(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(
                [
                    "timestamp",
                    "longitude",
                    "latitude",
                    "altitude",
                    "heading",
                    "pitch",
                    "roll",
                    f"sweep_seconds_{sweep_time_convention}",
                    "lidar_fpath",
                ]
            )
            for wgs84_row, lidar_row in lidars_with_wgs84:
                spamwriter.writerow(list(wgs84_row) + list(lidar_row))

        if check:
            # NOTE: The report may have duplicates, since an image may be missing a pose _AND_ be corrupted. The outputs
            # are not sorted by time or anything.
            with out_fs.open(report_fpath, "w") as csvfile:
                writer = csv.writer(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["lidar_uri", "timestamp", "status", "details"])
                for path, timestamp, message, details in stats:
                    writer.writerow([path, timestamp, message, details])

        report = ""
        report += "Date: " + datetime.isoformat(datetime.now()) + "\n"
        report += f"(log_root = {log_root})\n"
        report += f"{len(stats)} samples analyzed.\n"
        n_problems = len([x for x in stats if x[2] != "OK"])
        report += f"{n_problems} problems found."
        # for problem in status:
        #     report += "\t -" + problem + "\n"
        report += ""
        print(report)

        if check:
            print("\n\nWrote detailed health report to", report_fpath)
        else:
            print("Did not compute or dump detailed report.")



    def index_camera(
        self,
        log_id: str,
        cam_name: Union[CamName, str],
        out_index_fpath: Optional[str] = None,
        check: bool = False,
        pb_position: int = 0,
        debug_max_errors: int = 0,
        reindex: bool = False,
    ):
        """Please see `index_all_cameras` for info."""
        scheme = urlparse(self._root).scheme
        in_fs = fsspec.filesystem(scheme)
        if isinstance(cam_name, str):
            cam_name = CamName(cam_name)

        self._logger.info("Setting up log reader to process camera %s", cam_name.value)
        log_root = os.path.join(self._root, log_id.lstrip("/"))
        log_reader = LogReader(log_root_uri=log_root, map=self.map)
        cam_dir = os.path.join(log_root, "cameras", cam_name.value.lstrip("/"))

        # Collects diagnostics for writing the health report, if check is True. The diagnostics will NOT be sorted and
        # may contain duplicates.
        status = []

        # if out_index_fpath is None:
        #     out_index_fpath = os.path.join(log_root, "index", f"{cam_name}.geojson")
        if out_index_fpath is None:
            out_index_fpath = os.path.join(cam_dir, "index")

        regular_index_fpath = os.path.join(out_index_fpath, "index.csv")
        raw_wgs84_index_fpath = os.path.join(out_index_fpath, "raw_wgs84.csv")
        utm_index_fpath = os.path.join(out_index_fpath, "utm.csv")
        out_fs = fsspec.filesystem(urlparse(out_index_fpath).scheme)
        unindexed_fpath = os.path.join(out_index_fpath, "unindexed.csv")
        report_fpath = os.path.join(out_index_fpath, "report.csv")

        if in_fs.exists(report_fpath) and not reindex:
            print("Index already exists at", out_index_fpath)
            return

        # if not os.path.exists(os.path.join(log_root, "all_poses.npz.lz4")):
        #     continue

        # print(pose_times.shape)

        cp_dense = log_reader.continuous_pose_dense
        cp_times = cp_dense[:, 0]

        utm_poses = log_reader.utm_poses_dense
        mrp_poses = log_reader.map_relative_poses_dense
        # Check a dataset invariant as a smoke tests
        assert len(mrp_poses) == len(utm_poses)
        mrp_times = mrp_poses["time"]

        check_status = "deep checking" if check else "NO checking !!!"
        self._logger.info("Starting to index camera data. [%s]", check_status)
        index = []
        n_errors_so_far = 0

        progress_bar = tqdm(
            sorted(in_fs.glob(os.path.join(cam_dir, "*", "*.webp"))),
            mininterval=TQDM_MIN_INTERVAL_S,
            # Supposed to let each process have its own progress bar, but it doesn't work since the progress bars don't
            # start at the same time. Rip.
            position=pb_position,
            desc=f"{cam_name.value:<18}",
        )
        for entry in progress_bar:
            img_fpath = entry
            # XXX(andrei): Similarly, in a sibling method, look for large discrepancies between the number of samples
            # in different sensors. E.g., if one camera has 100 images and the other has 1000, that's a problem we would
            # like to look into.

            # XXX(andrei): Log a status error if meta is missing. May want to also quickly loop through metas and error
            # if no image for a specific meta.
            meta_fpath = entry.replace(".day", ".meta").replace(".night", ".meta").replace(".webp", ".npy")
            # Keep track of the log ID in the index, so we can merge indexes easily.
            img_fpath_in_root = "/".join(img_fpath.split("/")[-5:])
            progress_bar.set_postfix(n_errors_so_far=n_errors_so_far)
            if n_errors_so_far > debug_max_errors and debug_max_errors > 0:
                break

            if not in_fs.exists(meta_fpath):
                status.append((meta_fpath, timestamp_s, CAM_META_MISSING, f"Base entry uri: {entry}"))
                n_errors_so_far += 1
                continue

            timestamp_s = -1.0
            with in_fs.open(meta_fpath) as meta_f:
                try:
                    # The tolist actually extracts a dict...
                    meta = np.load(meta_f, allow_pickle=True).tolist()
                    timestamp_s = float(meta["capture_seconds"])
                    index.append(
                        (
                            timestamp_s,
                            img_fpath_in_root,
                            meta["shutter_seconds"],
                            meta["sequence_counter"],
                            meta["gain_db"],
                        )
                    )
                except UnpicklingError as err:
                    # TODO(andrei): Remove this hack once you re-extract with your ETL code
                    # hack for corrupted metadata, which should be fixed in the latest ETL
                    status.append((meta_fpath, timestamp_s, CAM_META_UNPICKLING_ERROR, str(err)))
                    n_errors_so_far += 1
                    continue
                except ModuleNotFoundError as err:
                    status.append((meta_fpath, timestamp_s, CAM_META_UNPICKLING_ERROR, str(err)))
                    n_errors_so_far += 1
                    continue
                    # TODO(andrei): Remove this one too
                    # seems like corrupted pickles can trigger this, oof
                    # err_msg = f"ERROR: ModuleNotFoundError reading metadata {str(err)}"
                    # status.append((meta_fpath, timestamp_s, err_msg))
                    # print(meta_fpath, err_msg)
                    # continue

            if check:
                try:
                    img_full_fpath = f"{scheme}://" + img_fpath
                    with in_fs.open(img_full_fpath, "rb") as img_f:
                        img = Image.open(img_f)
                        img.verify()
                        # This will actually read the image data!
                        img_np = np.asarray(img)
                        if img_np.shape != EXPECTED_IMAGE_SIZE:
                            status.append((img_full_fpath, timestamp_s, CAM_UNEXPECTED_SHAPE, str(img_np.shape)))
                            n_errors_so_far += 1
                            continue
                        else:
                            # Might indicate bad cases of over/underexposure. Likely won't trigger if the sensor is covered
                            # by snow (mean is larger than 5-10), which is fine since it's valid data.
                            img_mean = img_np.mean()
                            if img_mean < 5:
                                status.append((img_full_fpath, timestamp_s, CAM_MEAN_TOO_LOW, str(img_mean)))
                            elif img_mean > 250:
                                status.append((img_full_fpath, timestamp_s, CAM_MEAN_TOO_HIGH, str(img_mean)))
                            else:
                                status.append((img_full_fpath, timestamp_s, "OK", ""))
                except Exception as err:
                    status.append((img_full_fpath, timestamp_s, CAM_UNEXPECTED_CORRUPTION, str(err)))
                    n_errors_so_far += 1
                    continue

        # Sort by the capture time so we can easily search images by a timestamp
        image_index = sorted(index, key=lambda x: x[0])
        image_times = np.array([entry[0] for entry in image_index])

        # NOTE(andrei): WGS84 data is coarser, 10Hz, not 100Hz.
        self._logger.info("Reading WGS84 data")
        raw_wgs84_data = log_reader.raw_wgs84_poses_dense
        raw_wgs84_times = np.array(raw_wgs84_data[:, 0])

        imgs_with_pose = []
        imgs_with_wgs84 = []
        unindexed_frames = []
        # The 'max_delta_s' is just a logging thing. We will carefully inspect the deltas in the report analysis part,
        # but for now we only want WARNINGs to be printed if there's some serious problems. A couple of minutes of
        # missing poses (WGS84 or MRP-derived UTM) at the start of a log is expected (especially the localizer-based
        # MRP), and not a huge issue.
        utm_and_mrp_index = associate(image_times, mrp_times, max_delta_s=60 * 10)
        matched_timestamps = mrp_times[utm_and_mrp_index]
        deltas = np.abs(matched_timestamps - image_times)
        # invalid_times = deltas > 0.1
        for img_row, pose_idx, delta_s in zip(image_index, utm_and_mrp_index, deltas):
            img_fpath = img_row[1]
            img_time = float(img_row[0])
            if delta_s > 0.1:
                # error = f"WARNING: {img_time = } does not have a valid pose in this log [{delta_s = }]"
                status.append((img_fpath, img_time, CAM_UNMATCHED_POSE, str(delta_s)))
                unindexed_frames.append(img_row)
            else:
                imgs_with_pose.append((utm_poses[pose_idx, :], mrp_poses[pose_idx], img_row))

        raw_wgs84_index = associate(image_times, raw_wgs84_times, max_delta_s=60 * 10)
        matched_raw_wgs84_timestamps = raw_wgs84_times[raw_wgs84_index]
        raw_wgs84_deltas = np.abs(matched_raw_wgs84_timestamps - image_times)
        for img_row, wgs_idx, delta_wgs_s in zip(image_index, raw_wgs84_index, raw_wgs84_deltas):
            img_fpath = img_row[1]
            img_time = float(img_row[0])
            if delta_wgs_s > 0.5:
                # error = f"WARNING: {img_time = } does not have a valid raw WGS84 pose in this log [{delta_wgs_s = }]"
                status.append((img_fpath, img_time, CAM_UNMATCHED_RAW_WGS84, str(delta_wgs_s)))
            else:
                imgs_with_wgs84.append((raw_wgs84_data[wgs_idx, :], img_row))

        # TODO(andrei): Should we write CP? If we do we need clear docs, because using it will get you nonsense when
        # aggregating across logs.
        # with open(out_index_fpath, "w", newline="") as csvfile:
        #     spamwriter = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     spamwriter.writerow(["capture_time", "poses_and_differentials_valid", "x", "y", "z", "capture_seconds", "img_fpath_in_cam", "shutter_seconds", "sequence_counter", "gain_db"])
        #     for pose_row, img_row in imgs_with_pose:
        #         spamwriter.writerow(list(pose_row) + list(img_row))

        if not out_fs.exists(out_index_fpath):
            out_fs.mkdir(out_index_fpath)
        with out_fs.open(raw_wgs84_index_fpath, "w", newline="") as csvfile:
            spamwriter = csv.writer(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(
                [
                    "timestamp",
                    "longitude",
                    "latitude",
                    "altitude",
                    "roll",
                    "pitch",
                    "yaw",
                    "capture_seconds",
                    "img_fpath_in_cam",
                    "shutter_seconds",
                    "sequence_counter",
                    "gain_db",
                ]
            )
            for wgs84_row, img_row in imgs_with_wgs84:
                spamwriter.writerow(list(wgs84_row) + list(img_row))

        # Write a text file with all samples which could not be matched to an accurate pose.
        with out_fs.open(unindexed_fpath, "w", newline="") as csvfile:
            spamwriter = csv.writer(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["img_fpath_in_cam"])
            for entry in unindexed_frames:
                spamwriter.writerow([entry])

        with out_fs.open(utm_index_fpath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                [
                    "pose_timestamp",
                    "utm_e",
                    "utm_n",
                    "utm_alt",
                    "mrp_x",
                    "mrp_y",
                    "mrp_z",
                    "mrp_roll",
                    "mrp_pitch",
                    "mrp_yaw",
                    "mrp_submap_id",
                    "capture_seconds",
                    "img_fpath_in_cam",
                    "shutter_seconds",
                    "sequence_counter",
                    "gain_db",
                ]
            )
            for utm_pose, mrp_pose, img_row in imgs_with_pose:
                utm_e, utm_n = utm_pose
                # timestamp, valid, submap, x,y,z,roll,pitch,yaw = mrp_pose
                # TODO(andrei): Add UTM altitude here.
                writer.writerow(
                    [mrp_pose["time"], utm_e, utm_n, -1]
                    + [
                        mrp_pose["x"],
                        mrp_pose["y"],
                        mrp_pose["z"],
                        mrp_pose["roll"],
                        mrp_pose["pitch"],
                        mrp_pose["yaw"],
                        str(mrp_pose["submap_id"]),
                    ]
                    + list(img_row)
                )

        if check:
            # NOTE: The report may have duplicates, since an image may be missing a pose _AND_ be corrupted. The outputs
            # are not sorted by time or anything.
            with out_fs.open(report_fpath, "w") as csvfile:
                writer = csv.writer(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["img_fpath_in_cam", "timestamp", "status", "details"])
                for idx, (path, timestamp, message, details) in enumerate(status):
                    writer.writerow([path, timestamp, message, details])
                    if idx < 5:
                        print(f"{path} {timestamp} {message} {details}")

        report = ""
        report += "Date: " + datetime.isoformat(datetime.now()) + "\n"
        report += f"(log_root = {log_root})\n"
        report += f"{len(status)} potential problems found ({n_errors_so_far} errors):\n"
        # for problem in status:
        #     report += "\t -" + problem + "\n"
        report += ""
        print(report)

        if check:
            print("\n\nWrote detailed health report to", report_fpath)
        else:
            print("Did not compute or dump detailed report.")

        return report