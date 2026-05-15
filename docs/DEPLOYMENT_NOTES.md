# Deployment Notes: Modal + Cloudflare

This is a short personal checklist for publishing this FER project again.

## Current live setup

- Public URL: `https://fer.nishantvalvi.me`
- Compute and application hosting: Modal
- Public domain proxy: Cloudflare Worker (`fer-proxy`)
- Important: this is a **reverse proxy**, not a redirect. The browser must keep
  showing `fer.nishantvalvi.me`.

## Publish a new version on Modal

Run these commands from the repository root:

```powershell
modal deploy modal_app.py
```

Modal uploads the app code and prints a long `modal.run` URL. First open that
URL and test one image upload. The Cloudflare public URL will use the new
version automatically because the Modal app name stays the same.

If the Modal command is not available on a new computer:

```powershell
pip install modal
modal setup
```

Then sign in in the browser and run `modal deploy modal_app.py` again.

## Keep the short Cloudflare URL

Do **not** create a Redirect Rule. A redirect changes the browser address to
the long Modal URL.

For a new project/subdomain, use this pattern instead:

1. In **Cloudflare DNS**, create a proxied (orange-cloud) placeholder A record
   for the subdomain. Use `192.0.2.1` as the placeholder IP.
2. Create a Cloudflare Worker that forwards every request to the Modal URL.
   It must pass through the path and query string.
3. Add a Worker route such as `fer.nishantvalvi.me/*` to that Worker.
4. Open `https://fer.nishantvalvi.me`. The address bar must still show this
   short URL.
5. Test image upload first, then video, then webcam.

The current Worker is named `fer-proxy` and its route is
`fer.nishantvalvi.me/*`.

## Cost safety

- Keep Modal's spend limit at `$0` if you only want to use the monthly free
  credits.
- Do not create a Modal Volume for this demo unless you accept storage charges.
- Modal containers scale down after inactivity, so the first visit can be slow.

## Render files

`Dockerfile.render` and `render.yaml` were removed because this demo is now
published through Modal. The regular `Dockerfile` remains for local Docker use.
If Render is needed later, restore the old files from Git history rather than
maintaining two deployment configurations unnecessarily.
