# InfuseAtmosAdder (Alpha)

InfuseAtmosAdder provides a lightweight proxy that sits in between Jellyfin and Infuse, it can stitch an external Dolby DP Atmos track, or other audio files, into the video on the fly, without reencoding.

The original media on disk is never modified; Jellyfin keeps serving the video, and the proxy adds the matching audio track while the stream is running.

Not only that, but the TrueHD Atmos track is automatically replaced with the Infuse-compatible track, and set as default. Commentary and other such tracks are preserved.

### **[DeeZy](https://github.com/jessielw/DeeZy) effectively lets you play UHD Blu-Ray rips/remuxes with Atmos on Infuse!** *[More information here --->](https://community.firecore.com/t/help-get-more-dolby-atmos-on-apple-tv/16477/1303)*

Why such a hacky solution? Because Firecore haven't implemented external audio track support yet, [despite it being a core part of Jellyfin for years.](https://jellyfin.org/docs/general/server/media/movies#external-subtitles-and-audio-tracks)

Make sure to encourage the Infuse developers to add proper external audio file support [by liking and commenting here!](https://community.firecore.com/t/support-for-external-audio-files/15848) Hopefully this project get deprecated fast.


## Caveats

- Very Alpha: expect rough edges. Report issues with logs so they can be fixed.
- Progressive streaming: seeking behaves like a live stream because the proxy does not serve real byte ranges yet.
- Seeking doesn't work nearly at all, but skipping 10s forward a 100 times does
- File-system only: sidecars must live next to the video file; Jellyfin’s external-audio URLs are not used.

## Requirements

- A working Jellyfin server with media accessible on the filesystem.
- Infuse on Apple TV or iOS already configured to reach Jellyfin.
- Docker (Installation instructions differ per platform; there are plenty of walkthroughs availible online)
- Movie or episode files with external audio tracks that follow the naming pattern `<video name>.eac3` (or `.ec3`, `.ac3`).

> **Tip:** Run the proxy on the same machine that hosts Jellyfin whenever possible. That makes it easy to mount the same media folders and avoids cross-machine file-sharing headaches.

## Download the Project

Open a terminal on the machine that will run the proxy and run:

```bash
git clone https://github.com/kristofferR/InfuseAtmosAdder.git
cd InfuseAtmosAdder
```

If Git is not installed, install it first or download the ZIP from GitHub and extract it.

## Configure Media Paths

Edit `docker-compose.yml` with any text editor. Locate the `volumes:` section:

```yaml
    volumes:
      - /mnt/data/media/movies:/media/movies:ro
      - /mnt/data/media/TV:/media/TV:ro
      - /mnt/apps/InfuseAtmosAdder:/app
```

Each line follows the format `outside-path:inside-path[:ro]`:

- Text before the colon is the folder on your machine that holds the media files, only change that part.

Update the outside paths so they point to your local media directories. Keep the inside paths (`/media/...`) unchanged.

## Adjust Core Settings (Optional)

The defaults work for most setups. If needed, tweak these environment variables in `docker-compose.yml`:

- `JF_URL`: change the Jellyfin base URL if it differs from `http://10.0.0.50:8096`.
- `MEDIA_ROOTS`: confirm it lists the inside paths you mounted (colon-separated).
- `MUX_ON`: leave at `true` to enable remuxing. Set to `false` if you ever want to force passthrough.

See `AGENTS.md` for a full breakdown of advanced options ordered by how often people adjust them.

## Start the Proxy

From the project directory, launch the container in the background:

```bash
docker compose up -d
```

The first run downloads dependencies and prepares a virtual environment. When Docker finishes, the proxy listens on port `9999`.

To watch the startup logs:

```bash
docker compose logs -f
```

Look for uvicorn messages like `Uvicorn running on http://0.0.0.0:9999`.

## Connect Infuse

1. In Infuse, edit your Jellyfin share.
2. Change the server address to the host running the proxy (for example `http://proxy-hostname:9999`), or just add a dupe with the proxy port. If you add a new share with just a different port you can still use the old one exactly as before, you just get another one where external audio files works but skipping doesn't.
3. Keep your Jellyfin credentials unchanged.
4. Play a title that has a external audio track. Infuse should now list the extra audio stream.

## Updating or Stopping the Proxy

- Stop the container:

  ```bash
  docker compose down
  ```

- Pull updates:

  ```bash
  git pull
  docker compose pull
  docker compose up -d
  ```

## Troubleshooting Cheatsheet

- **Infuse only shows the original audio** – ensure the sidecar file name matches the video file (including punctuation) and uses a supported extension (`.ec3`, `.eac3`, `.ac3`).
- **Container refuses to start** – run `docker compose logs` and look for mount errors or typos in the `volumes` section.
- **Playback errors mid-stream** – leave `docker compose logs -f` running during playback and examine lines beginning with `[ffmpeg]` for codec issues.
- **Need to fine-tune behaviour** – refer to the configuration tables in `AGENTS.md`.



## Frequently Asked Questions


**Can this break Jellyfin?**

No. The proxy forwards the same requests and tokens that Infuse already uses. It does not alter your Jellyfin database or media.

**Does it work with other players?**

Non-Infuse clients fall back to straight passthrough, so they behave exactly as if Jellyfin were exposed directly.

---

Thanks for testing the alpha build! Open an issue if you run into problems, and include the relevant log snippet to speed up troubleshooting.

