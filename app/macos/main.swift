// Abliterator
// Copyright (C) 2026 David M. Berry
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The app: a window onto the instrument's page. On every launch it runs bootstrap.sh (which installs Python,
// the two environments and the model weights the first time, and checks them afterwards), then starts the
// server from the app's copy of the code and shows http://127.0.0.1:8775/. Quitting stops the server.
// If a server already answers on the port (one started from the source tree), the window shows that one.

import Cocoa
import WebKit

let port = Int(ProcessInfo.processInfo.environment["ABLITERATOR_APP_PORT"] ?? "") ?? 8775   // the override is for testing beside a running server
let base = URL(string: "http://127.0.0.1:\(port)/")!
let res = Bundle.main.resourceURL!
let src = res.appendingPathComponent("abliterator")
let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("Abliterator")
let logs = support.appendingPathComponent("logs")

final class App: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate, WKDownloadDelegate,
                 WKScriptMessageHandler {
    var window: NSWindow!
    var web: WKWebView!
    var server: Process?
    var setup: Process?
    var ownServer = false
    var quitting = false
    var activity: NSObjectProtocol?

    func applicationDidFinishLaunching(_ n: Notification) {
        // The engine plays in real time from a child process; App Nap and timer coalescing would starve it when
        // the window is hidden or in the background.
        activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .latencyCritical], reason: "Playing")
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        menus()
        let cfg = WKWebViewConfiguration()
        cfg.mediaTypesRequiringUserActionForPlayback = []
        cfg.userContentController.add(self, name: "setup")
        web = WKWebView(frame: .zero, configuration: cfg)
        web.navigationDelegate = self
        web.uiDelegate = self
        web.allowsMagnification = true
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1500, height: 940),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "Abliterator"
        window.contentView = web
        window.setFrameAutosaveName("Abliterator")
        if window.frame.origin == .zero { window.center() }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        alive { up in
            if up { self.web.load(URLRequest(url: base)) } else { self.runSetup() }
        }
    }

    // ---------------------------------------------------------------- first run, server
    func runSetup() {
        web.loadFileURL(res.appendingPathComponent("setup.html"), allowingReadAccessTo: res)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = [src.appendingPathComponent("app/bootstrap.sh").path, src.path, support.path,
                       res.appendingPathComponent("uv").path]
        p.environment = env()
        let out = Pipe()
        p.standardOutput = out
        p.standardError = out
        let log = handle("setup.log")
        var buf = Data()
        out.fileHandleForReading.readabilityHandler = { h in
            let d = h.availableData
            if d.isEmpty { return }
            log?.write(d)
            buf.append(d)
            while let i = buf.firstIndex(of: 10) {
                let line = String(decoding: buf[buf.startIndex..<i], as: UTF8.self)
                buf.removeSubrange(buf.startIndex...i)
                DispatchQueue.main.async { self.toPage(line) }
            }
        }
        p.terminationHandler = { p in
            out.fileHandleForReading.readabilityHandler = nil
            DispatchQueue.main.async {
                if p.terminationStatus == 0 { self.startServer() } else { self.toPage("{\"failed\": true}") }
            }
        }
        setup = p
        do { try p.run() } catch { toPage("{\"failed\": true, \"text\": \"\(error.localizedDescription)\"}") }
    }

    func toPage(_ line: String) {
        let json = line.hasPrefix("{") ? line : (try? String(data: JSONSerialization.data(withJSONObject: ["log": line]), encoding: .utf8)) ?? "{}"
        web.evaluateJavaScript("window.progress && window.progress(\(json))")
    }

    func startServer() {
        toPage("{\"step\": \"start\", \"state\": \"start\"}")
        let p = Process()
        p.executableURL = support.appendingPathComponent("env/server/bin/python")
        p.arguments = [src.appendingPathComponent("backend/server.py").path]
        p.environment = env()
        p.qualityOfService = .userInteractive
        p.currentDirectoryURL = support
        let log = handle("server.log")
        p.standardOutput = log
        p.standardError = log
        p.terminationHandler = { _ in
            DispatchQueue.main.async {
                if !self.quitting { self.toPage("{\"step\": \"start\", \"state\": \"failed\"}") }
            }
        }
        server = p
        ownServer = true
        do { try p.run() } catch { toPage("{\"step\": \"start\", \"state\": \"failed\"}"); return }
        wait(tries: 300)
    }

    func wait(tries: Int) {
        alive { up in
            if up { self.web.load(URLRequest(url: base)); return }
            if tries == 0 || self.server?.isRunning == false {
                self.toPage("{\"step\": \"start\", \"state\": \"failed\"}"); return
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { self.wait(tries: tries - 1) }
        }
    }

    func env() -> [String: String] {
        var e = ProcessInfo.processInfo.environment
        e["ABLITERATOR_DATA"] = support.appendingPathComponent("data").path
        e["ABLITERATOR_LOGS"] = logs.path
        e["ABLITERATOR_MRT2_PYTHON"] = support.appendingPathComponent("env/engine/bin/python").path
        e["ABLITERATOR_AUDIO_PORT"] = String(port)
        e["PYTHONDONTWRITEBYTECODE"] = "1"          // the app's files are signed and must not change
        e["PYTHONUNBUFFERED"] = "1"
        e["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        return e
    }

    func handle(_ name: String) -> FileHandle? {
        let u = logs.appendingPathComponent(name)
        if !FileManager.default.fileExists(atPath: u.path) { FileManager.default.createFile(atPath: u.path, contents: nil) }
        let h = try? FileHandle(forWritingTo: u)
        h?.seekToEndOfFile()
        return h
    }

    func alive(_ done: @escaping (Bool) -> Void) {
        var r = URLRequest(url: base.appendingPathComponent("api/status"))
        r.timeoutInterval = 1
        URLSession.shared.dataTask(with: r) { d, resp, _ in
            let ok = (resp as? HTTPURLResponse)?.statusCode == 200 && d != nil
            DispatchQueue.main.async { done(ok) }
        }.resume()
    }

    // The setup page's buttons: retry, open the logs folder.
    func userContentController(_ c: WKUserContentController, didReceive m: WKScriptMessage) {
        switch m.body as? String {
        case "retry": runSetup()
        case "logs": NSWorkspace.shared.open(logs)
        default: break
        }
    }

    // ---------------------------------------------------------------- quitting
    func applicationShouldTerminateAfterLastWindowClosed(_ a: NSApplication) -> Bool { true }

    func applicationShouldTerminate(_ a: NSApplication) -> NSApplication.TerminateReply {
        quitting = true
        setup?.terminate()
        guard ownServer, let p = server, p.isRunning else { return .terminateNow }
        var r = URLRequest(url: base.appendingPathComponent("api/stop"))
        r.httpMethod = "POST"
        r.setValue("1", forHTTPHeaderField: "X-Abliterator")
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = "{}".data(using: .utf8)
        r.timeoutInterval = 3
        URLSession.shared.dataTask(with: r) { _, _, _ in
            p.terminate()
            DispatchQueue.global().async {
                for _ in 0..<50 where p.isRunning { usleep(100_000) }
                if p.isRunning { kill(p.processIdentifier, SIGKILL) }
                DispatchQueue.main.async { a.reply(toApplicationShouldTerminate: true) }
            }
        }.resume()
        return .terminateLater
    }

    // ---------------------------------------------------------------- page: links, downloads, dialogs
    func webView(_ w: WKWebView, decidePolicyFor a: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if a.shouldPerformDownload { decisionHandler(.download); return }
        if let u = a.request.url, let s = u.scheme, s.hasPrefix("http"), u.host != "127.0.0.1" {
            NSWorkspace.shared.open(u)
            decisionHandler(.cancel); return
        }
        decisionHandler(.allow)
    }

    func webView(_ w: WKWebView, decidePolicyFor r: WKNavigationResponse, decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        decisionHandler(r.canShowMIMEType ? .allow : .download)
    }

    func webView(_ w: WKWebView, navigationAction: WKNavigationAction, didBecome d: WKDownload) { d.delegate = self }
    func webView(_ w: WKWebView, navigationResponse: WKNavigationResponse, didBecome d: WKDownload) { d.delegate = self }

    func download(_ d: WKDownload, decideDestinationUsing r: URLResponse, suggestedFilename name: String,
                  completionHandler: @escaping (URL?) -> Void) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = name
        panel.directoryURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
        panel.beginSheetModal(for: window) { ok in
            guard ok == .OK, let u = panel.url else { completionHandler(nil); return }
            try? FileManager.default.removeItem(at: u)
            completionHandler(u)
        }
    }

    func webView(_ w: WKWebView, createWebViewWith c: WKWebViewConfiguration, for a: WKNavigationAction,
                 windowFeatures f: WKWindowFeatures) -> WKWebView? {
        if let u = a.request.url { NSWorkspace.shared.open(u) }
        return nil
    }

    func webView(_ w: WKWebView, runJavaScriptAlertPanelWithMessage m: String, initiatedByFrame f: WKFrameInfo,
                 completionHandler: @escaping () -> Void) {
        let al = NSAlert(); al.messageText = m
        al.beginSheetModal(for: window) { _ in completionHandler() }
    }

    func webView(_ w: WKWebView, runJavaScriptConfirmPanelWithMessage m: String, initiatedByFrame f: WKFrameInfo,
                 completionHandler: @escaping (Bool) -> Void) {
        let al = NSAlert(); al.messageText = m
        al.addButton(withTitle: "OK"); al.addButton(withTitle: "Cancel")
        al.beginSheetModal(for: window) { r in completionHandler(r == .alertFirstButtonReturn) }
    }

    func webView(_ w: WKWebView, runJavaScriptTextInputPanelWithPrompt m: String, defaultText t: String?,
                 initiatedByFrame f: WKFrameInfo, completionHandler: @escaping (String?) -> Void) {
        let al = NSAlert(); al.messageText = m
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 280, height: 24))
        field.stringValue = t ?? ""
        al.accessoryView = field
        al.addButton(withTitle: "OK"); al.addButton(withTitle: "Cancel")
        al.window.initialFirstResponder = field
        al.beginSheetModal(for: window) { r in completionHandler(r == .alertFirstButtonReturn ? field.stringValue : nil) }
    }

    func webView(_ w: WKWebView, runOpenPanelWith p: WKOpenPanelParameters, initiatedByFrame f: WKFrameInfo,
                 completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = p.allowsMultipleSelection
        panel.beginSheetModal(for: window) { r in completionHandler(r == .OK ? panel.urls : nil) }
    }

    // ---------------------------------------------------------------- menus
    @objc func reload() { web.reload() }
    @objc func showData() { NSWorkspace.shared.open(support.appendingPathComponent("data")) }
    @objc func showLogs() { NSWorkspace.shared.open(logs) }
    @objc func site() { NSWorkspace.shared.open(URL(string: "https://vector-lab-tools.github.io/abliterator/")!) }
    // The standard About box with Credits.html (the About tab's text) as its credits. HTML comes in black, so the
    // text is given the label colour, which follows light and dark; links keep the link colour.
    @objc func about() {
        var opts: [NSApplication.AboutPanelOptionKey: Any] = [:]
        if let d = try? Data(contentsOf: res.appendingPathComponent("Credits.html")),
           let s = try? NSMutableAttributedString(data: d, options: [.documentType: NSAttributedString.DocumentType.html,
                                                                       .characterEncoding: String.Encoding.utf8.rawValue],
                                                  documentAttributes: nil) {
            let all = NSRange(location: 0, length: s.length)
            s.addAttribute(.foregroundColor, value: NSColor.labelColor, range: all)
            s.enumerateAttribute(.link, in: all) { v, r, _ in if v != nil { s.addAttribute(.foregroundColor, value: NSColor.linkColor, range: r) } }
            opts[.credits] = s
        }
        NSApp.orderFrontStandardAboutPanel(options: opts)
        NSApp.activate(ignoringOtherApps: true)
    }

    func menus() {
        let main = NSMenu()
        func add(_ title: String, _ items: [NSMenuItem]) {
            let m = NSMenu(title: title)
            items.forEach { m.addItem($0) }
            let it = NSMenuItem(); it.submenu = m; main.addItem(it)
        }
        func item(_ t: String, _ s: Selector?, _ k: String = "", _ target: AnyObject? = nil, _ mods: NSEvent.ModifierFlags = .command) -> NSMenuItem {
            let i = NSMenuItem(title: t, action: s, keyEquivalent: k); i.target = target; i.keyEquivalentModifierMask = mods; return i
        }
        add("Abliterator", [item("About Abliterator", #selector(about), "", self), .separator(),
                            item("Hide Abliterator", #selector(NSApplication.hide(_:)), "h"),
                            item("Hide Others", #selector(NSApplication.hideOtherApplications(_:)), "h", nil, [.command, .option]),
                            .separator(), item("Quit Abliterator", #selector(NSApplication.terminate(_:)), "q")])
        add("Edit", [item("Undo", Selector(("undo:")), "z"), item("Redo", Selector(("redo:")), "Z"), .separator(),
                     item("Cut", #selector(NSText.cut(_:)), "x"), item("Copy", #selector(NSText.copy(_:)), "c"),
                     item("Paste", #selector(NSText.paste(_:)), "v"), item("Select All", #selector(NSText.selectAll(_:)), "a")])
        add("View", [item("Reload", #selector(reload), "r", self), .separator(),
                     item("Actual Size", #selector(WKWebView.resetZoom(_:)), "0"),
                     item("Zoom In", #selector(WKWebView.zoomIn(_:)), "+"), item("Zoom Out", #selector(WKWebView.zoomOut(_:)), "-"),
                     .separator(), item("Enter Full Screen", #selector(NSWindow.toggleFullScreen(_:)), "f", nil, [.command, .control])])
        add("Window", [item("Minimize", #selector(NSWindow.performMiniaturize(_:)), "m"),
                       item("Zoom", #selector(NSWindow.performZoom(_:)))])
        add("Help", [item("Abliterator Website", #selector(site), "", self), .separator(),
                     item("Show Data Folder", #selector(showData), "", self), item("Show Logs", #selector(showLogs), "", self)])
        NSApp.mainMenu = main
    }
}

extension WKWebView {
    @objc func resetZoom(_ s: Any?) { pageZoom = 1 }
    @objc func zoomIn(_ s: Any?) { pageZoom = min(2, pageZoom + 0.1) }
    @objc func zoomOut(_ s: Any?) { pageZoom = max(0.5, pageZoom - 0.1) }
}

let app = NSApplication.shared
let delegate = App()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
