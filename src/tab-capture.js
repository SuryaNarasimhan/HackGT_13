(function () {
  window.TabCapture = {
    create(api, onStream, onError) {
      let peer = null, timer = null, disconnected = null, session = null;
      function stop() {
        session = null; clearTimeout(timer); clearTimeout(disconnected);
        if (peer) { peer.ontrack = peer.onconnectionstatechange = null; peer.close(); peer = null; }
      }
      api.onTabSignal(async message => {
        if (!session || message.session !== session || message.type !== 'offer' || peer) return;
        const id = session;
        const pc = peer = new RTCPeerConnection({ iceServers: [] });
        function fail(text) { if (peer === pc) { stop(); onError(text); } }
        timer = setTimeout(() => fail('Browser video did not connect. Reconnect the extension.'), 25000);
        pc.onconnectionstatechange = () => {
          if (pc.connectionState === 'failed') fail('Browser video connection failed. Reconnect the extension.');
          if (pc.connectionState === 'disconnected') disconnected = setTimeout(() => fail('Browser video disconnected.'), 5000);
          if (pc.connectionState === 'connected') { clearTimeout(timer); clearTimeout(disconnected); }
        };
        pc.ontrack = event => {
          if (event.track.kind !== 'video') return;
          const media = event.streams[0];
          if (!media) return fail('Browser sent no video stream.');
          event.track.onended = () => fail('The browser stopped sharing the call tab.');
          onStream(media).catch(() => fail('Could not display the browser video. Reconnect the extension.'));
        };
        try {
          await pc.setRemoteDescription({ type: 'offer', sdp: message.sdp });
          await pc.setLocalDescription(await pc.createAnswer());
          await new Promise((resolve, reject) => {
            if (pc.iceGatheringState === 'complete') return resolve();
            const limit = setTimeout(() => { pc.removeEventListener('icegatheringstatechange', changed); reject(new Error('ICE timeout')); }, 8000);
            function changed() { if (pc.iceGatheringState === 'complete') { clearTimeout(limit); pc.removeEventListener('icegatheringstatechange', changed); resolve(); } }
            pc.addEventListener('icegatheringstatechange', changed);
          });
          if (peer !== pc || session !== id) return;
          await api.tabAnswer({ session: id, type: 'answer', sdp: pc.localDescription.sdp });
        } catch { fail('Could not negotiate browser video. Reconnect the extension.'); }
      });
      return {
        async start(audio) {
          stop();
          const result = await api.startTabCapture(audio);
          session = result.session;
          return result;
        },
        stop
      };
    }
  };
})();
