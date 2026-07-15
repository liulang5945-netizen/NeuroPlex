const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  const outDir = 'E:/taiji/.taiji_test_tmp';
  fs.mkdirSync(outDir, { recursive: true });

  const shots = [
    { url: 'http://localhost:5173/', file: 'fe-chat-desktop.png', wait: 9000 },
    { url: 'http://localhost:5173/#/life', file: 'fe-life-desktop.png', wait: 1500 },
    { url: 'http://localhost:5173/#/train', file: 'fe-train-desktop.png', wait: 1500 },
    { url: 'http://localhost:5173/#/workspace', file: 'fe-workspace-desktop.png', wait: 1500 },
    { url: 'http://localhost:5173/#/settings', file: 'fe-settings-desktop.png', wait: 1500 },
    { url: 'http://localhost:5173/#/kb', file: 'fe-kb-desktop.png', wait: 1500 },
    { url: 'http://localhost:5173/#/agent', file: 'fe-agent-desktop.png', wait: 1500 },
  ];
  for (const s of shots) {
    try {
      await page.goto(s.url, { waitUntil: 'domcontentloaded', timeout: 15000 });
    } catch (e) { console.log('goto warn', s.url, e.message); }
    await page.waitForTimeout(s.wait);
    await page.screenshot({ path: path.join(outDir, s.file), fullPage: false });
    console.log('saved', s.file);
  }

  // mobile viewport
  await ctx.close();
  const mctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true });
  const mpage = await mctx.newPage();
  try { await mpage.goto('http://localhost:5173/', { waitUntil: 'domcontentloaded', timeout: 15000 }); } catch(e){}
  await mpage.waitForTimeout(2000);
  await mpage.screenshot({ path: path.join(outDir, 'fe-chat-mobile.png'), fullPage: false });
  console.log('saved fe-chat-mobile.png');

  await browser.close();
  console.log('DONE');
})().catch(e => { console.error('FATAL', e); process.exit(1); });


