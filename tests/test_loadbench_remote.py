import binascii
import io
import struct
import unittest
from unittest.mock import patch
import zipfile
from src.remote_zip import HTTPRangeReader
from src.loadbench_remote import read_member,decode_stored_window


class Response(io.BytesIO):
    status=206
    def __init__(self,body,content_range):
        super().__init__(body);self.headers={'Content-Range':content_range}


class RemoteTests(unittest.TestCase):
    def test_range_must_match_requested_bytes(self):
        r=HTTPRangeReader('https://example.invalid/a.zip',100)
        with patch('urllib.request.urlopen',return_value=Response(b'abcd','bytes 10-13/100')):
            self.assertEqual(r.fetch(10,14),b'abcd')
        with patch('urllib.request.urlopen',side_effect=lambda *a,**k:Response(b'abcd','bytes 0-3/100')),patch('time.sleep'):
            with self.assertRaises(ValueError):r.fetch(10,14)
        self.assertEqual(r.bytes_downloaded,4)

    def test_no_accidental_whole_archive_download(self):
        response=Response(b'entire archive','');response.status=200
        with patch('urllib.request.urlopen',side_effect=lambda *a,**k:response),patch('time.sleep'):
            with self.assertRaises(ValueError):HTTPRangeReader('https://example.invalid',100).fetch(0,4)

    def test_complete_member_crc_checked(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('params.cir',b'.param w=1u')
        raw=data.getvalue()
        class Memory:
            def fetch(self,start,end):return raw[start:end]
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            m=z.infolist()[0]
            spec={'name':m.filename,'header_offset':m.header_offset,'compressed_size':m.compress_size,
                  'size':m.file_size,'crc32':m.CRC,'method':m.compress_type}
        self.assertEqual(read_member(Memory(),spec),b'.param w=1u')
        spec['crc32']^=1
        with self.assertRaises(ValueError):read_member(Memory(),spec)

    def test_stored_wrapper_partial_and_complete(self):
        blocks=[b'A'*100,b'B'*101,b'C'*102,b'D'*103]
        encoded=b''
        for i,body in enumerate(blocks):
            encoded+=bytes([int(i==3)])+struct.pack('<HH',len(body),len(body)^65535)+body
        self.assertEqual(decode_stored_window(encoded,True),b''.join(blocks))
        self.assertEqual(decode_stored_window(encoded[:-7]),b''.join(blocks)[:-7])
        with self.assertRaises(ValueError):decode_stored_window(b'not a stored deflate stream')


if __name__=='__main__':unittest.main()
