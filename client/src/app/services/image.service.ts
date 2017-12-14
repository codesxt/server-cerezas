import { Injectable } from '@angular/core';
import { Http, Headers, RequestOptions, Response } from '@angular/http';
import 'rxjs/add/operator/map';
import { environment } from '../../environments/environment';

@Injectable()
export class ImageService {
  baseURL: string = environment.apiUrl;
  constructor(
    private http: Http
  ) { }

  uploadImage(formData){
    let headers = new Headers({

    });
    let options = new RequestOptions({
      headers: headers
    });
    return this.http.post(this.baseURL+'/api/v1/upload', formData, options).map(
      (response: Response) =>response.json()
    )
  }
}
